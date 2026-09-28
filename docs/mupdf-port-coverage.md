# MuPDF port coverage map (`kopitiam-pdf`)

* **Date:** 2026-09-28 (SGT)
* **Upstream:** MuPDF `source/fitz/*.c` + `source/pdf/*.c` at pinned commit **`19f1284`** (AGPL-3.0, © Artifex), vendored read-only at `crates/kopitiam-pdf/vendor/mupdf/`
* **Port:** `crates/kopitiam-pdf/src/mupdf/*.rs` (61 files, 43,911 lines) plus `src/mupdf_extract.rs`
* **Produced by:** a read-only audit (AI, Claude). No code was changed. This doc and `docs/port-ledger.md` answer different questions: the ledger lists *which Rust file cites which C file*. This doc lists *which C behaviour actually runs, and what the user sees when it does not*.

**Snapshot caveat, hor.** (0.4.2: the draw-affine and draw-scale rows below were indeed out of date; both are translated now -- see the 0.4.2 section.) The audit read the working tree as it stood on 2026-09-28. While it was running, `src/mupdf/draw_affine.rs` and `src/mupdf/draw_scale.rs` appeared as new untracked files, and `mod.rs` was modified. That is someone else's in-flight work, **not reflected here**. So the `draw-affine.c` and `draw-scale-simple.c` rows (nearest-neighbour only, no down-scaling filter) may already be out of date. Re-check those two before quoting them.

This is a map, not a verdict lah. Every "missing" row below says what breaks on screen or in extracted text, so whoever picks the next wave can rank by what users actually hit, not by line count.

## 0.4.2 status -- read this first (updated 2026-09-28, after the translation tranches)

The audit below is a **snapshot taken before** the 0.4.2 translation work.
It is kept as it was written, so a reader can see what each claim said and
why it shaped the tranche order. Every row the tranches changed carries a
**0.4.2** note, and the old text is struck through beside it. Each gap was
closed with a synthetic regression test that fails on the old tree, and was
checked against real mutool from the pinned commit. Methods and numbers:
[`mupdf-code-to-code.md`](mupdf-code-to-code.md).

**File counts per status, now** (the audit's snapshot counts are further down):

| Status | Snapshot | 0.4.2 | Moved |
|---|---:|---:|---|
| translated | 26 | **37** | +pdf-repair, pdf-type3, pdf-function, pdf-pattern, pdf-shade, draw-scale-simple, draw-mesh, shade, compressed-buffer, draw-affine, draw-blend |
| partial | 39 | **38** | +pdf-layer; -compressed-buffer, -draw-affine |
| substituted-by-design | 19 | **22** | +filter-jbig2 (hayro-jbig2), load-jpx (hayro-jpeg2000), crypt-sha2 (`sha2`) |
| missing | 29 | **16** | the 13 files above (12 + draw-blend) left this bucket |
| out-of-scope | 92 | **92** | -- |

**The audit's top gaps, where they stand:**

| # | Gap (audit ranking) | Now |
|---|---|---|
| 1 | Broken-file repair; `endstream` scan for a bad `/Length` | **Closed.** `repair.rs` + the endstream filter. |
| 2 | AES-256 (`/R 5`, `/R 6`) refused | **Closed.** Ported from pdf-crypt.c. R2-R4 owner-password auth and a password API remain. |
| 3 | JBIG2 / JPX images blank | **Closed.** Via hayro-jbig2 / hayro-jpeg2000. |
| 4 | Shadings absent | **Closed.** All 7 types, pixel-identical on the feature corpus. |
| 5 | `gs` ignored (alpha, blend modes, soft masks, LW/D/Font) | **Closed.** Alpha, LW/D/Font and line state (tranche 3); blend modes, soft masks and transparency groups (tranche 9). Knockout is drawn non-knockout. |
| 6 | Clipping bbox-only | **Closed.** Exact clip masks, text clip, Form `/BBox`. |
| 7 | Stencil masks / `/Mask` | **Closed.** Stencils, colour keys, `/Mask` streams. |
| 8 | Inline images skipped | **Closed.** |
| 9 | Tiling patterns missing | **Closed.** Both `pdf_show_pattern` branches, including the tile cache's integer placement. |
| 10 | Colour spaces approximated | **Closed** for MuPDF's no-ICC path (Indexed, Separation/DeviceN, Lab). ICC itself is still missing. |
| next tier | CropBox, optional content, line style, Type3 | **Closed.** CropBox was a maintainer-reported bug ("my iaea tecdoc document is unusually wide"). |

**What remains, ranked by what a reader hits:**

1. ~~**ActualText** (`BDC /ActualText`): the one open-corpus text page that
   still differs (NUREG/CR-7289 p. 2).~~ **Closed in 0.4.2**: ported, including
   the lookup of the structure element via `/MCID` (that page's ActualText
   sits on the structure elements). All 1368 open-corpus text pages now match.
   Left: a text span ends at a subset of MuPDF's `pdf_flush_text` triggers,
   and the Alt/E/T metatexts are not forwarded (they do not change stext).
2. **No ICC colour management** (`color-lcms.c`) and **no overprint/spot
   simulation** (`separation.c`). The oracle runs `mutool -N -M 0` for
   exactly this reason. A default mutool render differs from ours on CMYK
   and spot pages.
3. **Named CJK CMaps** (`cmaps/*.h`) are not ported, and neither are the
   Adobe `*-UCS2` collection maps. A CJK document with a non-Identity
   encoding is misparsed. **Vertical writing** (`/W2`, stext wmode) and
   **bidi** are missing too.
4. **Non-embedded CJK/script fonts** have no Noto fallback faces (hayro
   covers the screen).
5. **Annotations without `/AP`** other than Ink are not synthesised.
   **Annotations are not in extracted text.**
6. **Page labels**, **R2-R4 owner-password auth / a password API**,
   **`/Matte`**, **colour keys on DCT/JPX images**.
7. Speed only: the glyph cache, the display list, and the image cache.
8. Glyph shape: the non-embedded base-14 faces are hayro's bundled fonts,
   not MuPDF's Nimbus. Measured cost is about 0.9-1.8 % gross on 48-70 pt
   bold text.

## How this was generated (method, so you can re-run it and argue with it)

Two layers: a mechanical count, then a manual read of the Rust to set the status.

**1. Mechanical count (appendix table).** A throwaway Python script (in the session scratchpad, not committed) did this:

* Enumerate every `*.c` in `vendor/mupdf/source/fitz` (149 files) and `vendor/mupdf/source/pdf` (56 files). That is 205 files, 191,030 lines. The two `.cpp` files (`tessocr.cpp`, `zxingbarcode.cpp`) and the `.h` tables are not counted: they are OCR/barcode glue plus data, all out of scope.
* **Function definitions.** A line at column 0 that is not `#`, `{`, `}`, a comment, whitespace, `typedef` or `enum`, and that contains `identifier(`. It counts as a definition only if a line starting with `{` comes within 25 lines and before any line ending in `;`. This covers both MuPDF styles: return type on its own line (`fz_transform_rect(...)`), and one-line `static void foo(...)`. C keywords are excluded. Names are de-duplicated per file. Total: **6,380 functions**.
* **"Named in kopitiam-pdf".** Tokenise every `.rs` under `crates/kopitiam-pdf/src/` (so `mupdf/`, `gui_frontend/`, `bin/` and the top-level files).
  * A C name with an `fz_`/`pdf_`/`ucdn_` prefix counts as referenced if it appears anywhere as a token. That catches a breadcrumb like `// MuPDF: fz_transform_rect (geometry.c:519)` or any prose mention.
  * A name without a prefix (a `static` helper like `next_flated`, `paeth`, `iswhite`) counts only if it appears in a Rust file that **also names this C file by basename** (e.g. `filter-predict.c`). This attribution rule is there because the first pass, which accepted any token match, gave false hits. `show_char` was credited to `pdf-subset.c` and `pdf-op-vectorize.c`. `width`, `box` and `triangle` were credited to `draw-scale-simple.c`. `step` and `clean` were credited to `warp.c`.
* Result: **593 of 6,380** C functions are named somewhere in the port (9.3%). Of the in-scope files (everything not `out-of-scope`), it is **589 of 4,212** (14.0%).

**How to read the %.** It is a floor, not a coverage score. The port renames and fuses functions freely (AID-0051: free functions become methods, the C helpers collapse into Rust idioms). So a translated file can score 40% while its behaviour is fully there. Two cases in point: `pdf-cmap.c`'s splay tree became a sorted `Vec`, and `stream-read.c`'s refill machinery became an in-memory window. The opposite error is small. After the attribution rule, the `out-of-scope` bucket gets only 4 hits and the `missing` bucket gets 1 hit (the `pdf_is_ocg_hidden` mention in `annot_run.rs`'s "not modelled" note). So a non-zero % always means someone at least wrote the name down.

| Status bucket | Files | C functions | Named in port | % |
|---|---:|---:|---:|---:|
| translated | 26 | 761 | 315 | 41.4 |
| partial | 39 | 2,531 | 256 | 10.1 |
| substituted-by-design | 19 | 324 | 17 | 5.2 |
| missing | 29 | 596 | 1 | 0.2 |
| out-of-scope | 92 | 2,168 | 4 | 0.2 |
| stubbed | 0 | – | – | – |
| **total** | **205** | **6,380** | **593** | **9.3** |

**2. Manual status.** Each file's status comes from reading the Rust module that claims to port it, **not** from its doc comment. Several doc comments were found to be out of date; they are listed in [Doc claims the code contradicts](#doc-claims-the-code-contradicts). The statuses mean:

* `translated`: the file's PDF-input behaviour is ported.
* `partial`: some of it is ported; the row says what is missing.
* `stubbed`: recognised but ignored. **No whole file is stubbed**, but many *features* are (parsed-and-ignored operators). See the [feature-level table](#feature-level-breakdown-content-stream-operators-and-render-features).
* `substituted-by-design`: MuPDF calls a C library (or its own runtime layer) and the port uses a pure-Rust crate or a Rust idiom instead. Per AID-0052 for FreeType/libjpeg/zlib, and AID-0051 for `fz_context`, memory and exceptions.
* `missing`: in scope, not ported.
* `out-of-scope`: not part of reading, rendering or extracting PDF input.

Every "missing" and "partial" claim in the main table was checked against the Rust source; file:line pointers are given for the load-bearing ones.

**One thing to know before reading the consequences: the hayro fallback.** `mupdf::rasterize_page` (the name every caller uses) is `hayro_fallback::rasterize_page_graceful`. It re-renders the **whole page** with the `hayro` crate, but **only** when kopitiam's own engine painted at least one advance-box glyph (`draw_device.rs` `fallback_glyphs`). So:

* Font gaps (Type3, non-embedded CJK, undecodable programs) are usually *masked* on screen by hayro.
* Every other gap (shadings, JBIG2/JPX, transparency, clipping, patterns) is **not** masked. Those pages have no box glyphs, so the native render is what the user sees.
* Text extraction and `rasterize_page_native` never fall back.
* A document that fails `PdfDocument::open` cannot reach hayro either: the fallback takes an already-open `PdfDocument`.

## Summary

**File counts per status** (205 C files, **snapshot before 0.4.2** -- current counts are in the 0.4.2 section at the top):

| Status | fitz | pdf | Total |
|---|---:|---:|---:|
| translated | 18 | 8 | **26** |
| partial | 21 | 18 | **39** |
| substituted-by-design | 18 | 1 | **19** |
| missing | 19 | 10 | **29** |
| stubbed | 0 | 0 | **0** (feature-level stubs listed separately) |
| out-of-scope | 73 | 19 | **92** |

**Top gaps, ranked by user impact** (snapshot -- 0.4.2 status of each is in the table at the top; all closed). The ranking is the auditor's judgement of how often an ordinary reader's PDF hits each gap. It was not measured against a corpus. The orchestrator's code-to-code harness section below is where measurement goes.

1. **No broken-file repair (`pdf-repair.c`), and no `endstream` scan for a bad `/Length` (`fz_open_endstream_filter`).** A PDF with a damaged, truncated or wrongly-offset xref fails `PdfDocument::open` outright (`xref.rs:127-130`, "repair is deferred"). MuPDF would rebuild the xref and open it. Nothing renders, nothing extracts, and the hayro fallback cannot help because it needs an open document. Separately, a stream whose `/Length` is wrong is clamped (`xref.rs:782-786`) and never re-scanned for `endstream` the way MuPDF does (`pdf-stream.c:374`, `pdf-parse.c:960`). A too-long length decodes to an empty stream; a too-short one gets truncated. Either way the user sees a blank page or a page cut off halfway.
2. **AES-256 encryption (`/R 5`, `/R 6`) is refused** (`crypt.rs:27-28, :194`), and so is `crypt-sha2.c`'s SHA-2 key derivation. Acrobat X and later write AES-256 by default. So a modern owner-restricted form (empty user password), which MuPDF opens with no prompt, fails to open here. Also missing: owner-password authentication, the public-key (`/Adobe.PubSec`) handler, and any caller-supplied password (only the empty user password is tried, `xref.rs:151-157`).
3. **JBIG2Decode and JPXDecode images are not decoded** (`page_image.rs:140-141`; `filter-jbig2.c`, `load-jpx.c`). The draw path skips the image silently (`page_run.rs:286-289`). A scanned document stored as JBIG2, the dominant codec for archival and library scans, renders as a **blank white page**, and `page_full_image` returns an error, so the OCR fallback cannot run on it either. A figure stored as JPEG 2000 renders blank.
4. **Shadings are absent end to end**: `sh` is ignored, shading patterns and `/Pattern` type 2 are ignored, and `pdf-shade.c`, `shade.c`, `draw-mesh.c` and `pdf-function.c` are all missing. Gradient backgrounds, shaded chart bars, gradient logos and slide-deck backdrops simply **do not paint**, and whatever is underneath shows through (usually white).
5. **The ExtGState `gs` operator is parsed and ignored** (`interpret.rs:45`, falls to `_ => {}` at :497), so:
   * no constant alpha (`CA`/`ca`),
   * no blend modes (`draw-blend.c` missing),
   * no soft masks (`SMask` in ExtGState),
   * no `LW`/`D`/`Font` set through `gs`.

   A translucent highlight box, watermark or drop shadow paints **fully opaque** and can hide the text under it. This also hits kopitiam's own synthesised Ink annotations: `annot_appearance.rs:303-311` writes `/GS0 gs` for opacity, and the interpreter drops it, so a 50%-opacity ink stroke draws solid.
6. **Clipping is bounding-box only** (`op_run.rs:297-313`, TODO(draw)). A non-rectangular clip (a photo cropped to a circle, a chart clipped to a plot area with a curved edge, text used as a clip with `Tr 4-7`) paints **outside** its intended shape. The over-paint is up to the clip path's bounding rectangle. Form XObject `/BBox` clips are not applied at all (`page_run.rs:29-30`).
7. **Stencil image masks (`/ImageMask true`) and `/Mask` are not honoured as masks.** An image mask is decoded as an opaque gray image (`page_image.rs:372-375`) and blitted with no mask semantics. So instead of painting the current fill colour through the mask, it paints black where the mask marks and **opaque white everywhere else**, wiping out anything drawn beneath it. Coloured icons and logos drawn as stencils come out black. Colour-key `/Mask` arrays and explicit `/Mask` streams are ignored too (no `Mask` lookup anywhere in `page_image.rs`), so the "transparent" background colour of such an image shows as a solid rectangle. (Only `/SMask` soft masks on images are honoured, `draw_device.rs:302-318`.)
8. **Inline images (`BI … ID … EI`) are skipped, not decoded** (`interpret.rs:508-547`). Small inline bitmaps render blank: TeX-generated bitmap figures, logos embedded by some producers, scanned-page strips, and bitmap Type3 glyph procs.
9. **Tiling patterns (`pdf-pattern.c`) are missing.** `scn /P0` with no components is a silent skip, and components in a `/Pattern` space map to the previous colour (`op_run.rs:349-355`, `resources.rs:87`). A hatched or textured area paints as a **solid block in whatever colour was current before**, or not at all.
10. **Colour spaces beyond Device* are approximated or refused.** The full breakdown is in the pdf-colorspace/colorspace rows. In short:
    * Fills in `Separation`/`DeviceN` become a gray of `1 - max(tint)`, because the tint transform is never run.
    * `Indexed` fills use the raw index as a gray value, clamped (`resources.rs:86`), so almost every indexed fill paints **white**.
    * `Lab` fills are treated as RGB (`resources.rs:219`), so L\*=50 reads as red=50, clamps to 1, and paints saturated red.
    * No ICC management at all.
    * Images in `Separation`/`DeviceN`/`Lab` return an error and render blank (`page_image.rs:692-693`).

**Next tier** (real, but narrower):

* **`CropBox` is ignored.** The page transform and pixmap size use only the `MediaBox` (`page_run.rs:139-175`), while MuPDF intersects with the CropBox (`pdf-page.c:765-771, 805-809`). Pages from publishers and LaTeX show extra margins, printer's marks and bleed. Text outside the crop gets extracted.
* **Optional content (`pdf-layer.c`) and marked content are ignored.** Hidden layers (alternate-language layers, print-only watermarks, CAD layers switched off) render and extract. `/ActualText` is not honoured, so extraction returns the raw glyph Unicode instead of the author's replacement text.
* **Annotations are not in extracted text.** `page_to_stext` runs only `/Contents` (`stext_device.rs:649`); `fz_new_stext_page_from_page` in MuPDF runs annots too. Filled-in form values and FreeText comments are missing from extraction and search.
* **Named CJK CMaps (`pdf-cmap-load.c`'s `cmaps/*.h`) are not ported.** Only Identity-H/V exist (`cmap.rs:309-315`). Any other named encoding falls back to Identity-H (`font.rs:367-371`), so a 1-byte/2-byte mixed encoding like `90ms-RKSJ-H` is **misparsed into the wrong CIDs**. Extraction is garbled and advances are wrong.
* **Line style is not honoured.** `d` (dash), `J`, `j` and `M` are ignored. Every stroke uses round caps and joins (`draw_path.rs:227-235`), and dashed lines draw **solid**.
* **Type3 fonts are not executed** (`pdf-type3.c` missing; `d0`/`d1` ignored). The native render paints advance boxes, which does trigger the hayro whole-page fallback. For extraction, `/Widths` are taken verbatim without `/FontMatrix` scaling (`font.rs:46-47`), so a Type3 font whose FontMatrix is not 0.001 gives wrong glyph advances: spurious or missing spaces.
* **Vertical writing (`wmode 1`) is stubbed.** No `/W2`/`/DW2` metrics, and the stext vertical arm is stubbed (`stext_device.rs:47-49`). Vertical Japanese/Chinese text extracts in the wrong order with broken spacing.
* **Bidi is missing** (`bidi.c`). Arabic and Hebrew text extracts in visual rather than logical order.

## Main table

Legend for the Status column: T = translated, P = partial, D = substituted-by-design, M = missing, O = out-of-scope. Target paths are under `crates/kopitiam-pdf/src/mupdf/` unless stated.

### PDF layer: document, objects, xref, streams, crypt

| MuPDF source | Key functions | kopitiam-pdf target | Status | Consequence if not translated / notes |
|---|---|---|---|---|
| `pdf-lex.c` | `pdf_lex`, `lex_number`, `lex_string`, `lex_hex_string` | `lex.rs` | T | Faithful. 13/19 functions named. |
| `pdf-parse.c` | `pdf_parse_array`, `pdf_parse_dict`, `pdf_parse_ind_obj`, `pdf_parse_stm_obj` | `parse.rs` | T | **0.4.2 (2026-09-28):** the `pdf_parse_ind_obj` repair hook (`try_repair`) now feeds the ported `repair.rs`. _Audit text (snapshot, before 0.4.2):_ ~~Recovery for stray tokens is kept. The `pdf_parse_ind_obj` repair hooks feed `pdf-repair.c`, which does not exist here.~~ |
| `pdf-object.c` | `pdf_new_*`, `pdf_dict_get*`, `pdf_to_*`, `fmt_obj`/`fmt_str` | `object.rs` (read), `write.rs` (`fmt_*`) | T | Read-side accessors are complete. Refcounted and shared `pdf_obj` is replaced by an owned `Object` enum (`Clone` deep-copies). The dirty-flag, journal and undo machinery is not ported; `annot_edit.rs` has its own `EditHistory`. |
| `pdf-xref.c` | `pdf_load_xref`, `pdf_read_old_xref`, `pdf_read_new_xref`, `pdf_load_obj_stm`, `pdf_cache_object`, `pdf_open_document_with_stream` | `xref.rs` | P | **0.4.2 (2026-09-28):** repair on failure is ported and wired at open and mid-read (`pdf_init_document`, `pdf_cache_object`), with `load_xref`'s repair triggers and `check_xref_entry_offsets`; stream decryption uses the object's real generation. Still missing: linearization/progressive loading, local/journal xrefs. _Audit text (snapshot, before 0.4.2):_ ~~**Ported:** classic tables, xref streams, hybrid `/XRefStm`, the `/Prev` chain (so incremental updates *read* correctly), object streams, and page-tree inheritance. **Missing:** repair on failure (see `pdf-repair.c`); linearization and progressive loading, which makes no visible difference for a local file; the `pdf_xref_entry` generation check; and local/journal xrefs. A document whose `startxref` or an xref offset is wrong **fails to open**.~~ |
| `pdf-repair.c` | `pdf_repair_xref_base`, `pdf_repair_obj`, `pdf_repair_obj_stms` | ~~none~~ `repair.rs` | ~~**M**~~ **T** | **0.4.2 (2026-09-28):** ported in full (`repair.rs`): the scan loop, `pdf_repair_obj`, object-stream recovery (incl. Bug 708286), `pdf_repair_roots`/`pdf_repair_trailer`, wired where MuPDF wires it. Code-to-code verified on 17 damaged files. Divergences (no bias/linearization/FDF) are documented in the module. _Audit text (snapshot, before 0.4.2):_ ~~**Highest impact.** Damaged, truncated or hand-edited PDFs (a download cut short, an xref shifted by a bad editor) fail with a format error. MuPDF scans for `N G obj` and rebuilds. The user sees "cannot open" where every other viewer shows the document.~~ |
| `pdf-stream.c` | `pdf_open_stream`, `pdf_open_filter`, `build_filter`, `pdf_load_stream` | `doc_stream.rs`, `xref.rs` `open_stream`/`stream_raw_num` | P | **0.4.2 (2026-09-28):** the endstream-scanning raw filter is ported (`fz_open_endstream_filter` semantics): a wrong `/Length` reads to `endstream` as in MuPDF. Still missing: named `/Crypt` filters per stream, `/F` external streams. _Audit text (snapshot, before 0.4.2):_ ~~The filter chain wiring and `/DecodeParms` pairing are ported. **Missing:** the `endstream`-scanning raw filter (`pdf-stream.c:374`), so a wrong `/Length` gives an empty or truncated stream (blank or half-drawn page) where MuPDF recovers; the `/Crypt` filter with named crypt filters per stream; and `/F` external file streams (these are rare).~~ |
| `pdf-crypt.c` | `pdf_new_crypt`, `pdf_authenticate_password`, `pdf_compute_object_key`, `pdf_open_crypt` | `crypt.rs` (from ISO 32000 §7.6, cross-checked against pdf-crypt.c), `xref.rs` `build_decryptor`/`rewrite_decrypted` | P | **0.4.2 (2026-09-28):** R5/R6 AES-256 are ported (`pdf_compute_encryption_key_r5/_r6`, the hardened hash, `/AESV3`), with R5/R6 owner-password authentication and MuPDF's rule that an empty password never opens as owner. Still missing: R2-R4 owner-password auth, a caller password API, PubSec, encrypt-on-write. _Audit text (snapshot, before 0.4.2):_ ~~**Ported:** the Standard handler at R2/R3/R4 (RC4-40, RC4-128, AES-128 via `/CF`); `EncryptMetadata`; and the `/Identity` filter. **Missing:** R5/R6 AES-256 (**owner-restricted forms from Acrobat X+ fail to open**); owner-password auth; user passwords other than empty (no API); the PubSec handler; and encrypt-on-write. At open, the whole document is rewritten as plaintext, so a save writes a decrypted copy without the owner restrictions (documented at `xref.rs:161-176`). See also the gen-0 finding under [Surprises](#surprises-and-defects-found-in-passing).~~ |
| `pdf-nametree.c` | `pdf_lookup_name`, `pdf_lookup_dest`, `pdf_load_name_tree` | `destination.rs` (`/Dests` name tree + PDF 1.1 `/Dests` dict, `/Limits` descent) | P | Destination lookup works. There is no generic name-tree API, so `/EmbeddedFiles` and `/JavaScript` name trees are not enumerable. User-visible: no attachment list. |
| `pdf-store.c` | `pdf_store_item`, `pdf_find_item` | per-`Processor` font `HashMap` (`resources.rs` `op_tf`); object cache in `xref.rs` | D | Idiomatic replacement. No cross-page resource cache for images, so a large image repeated on every page is re-decoded per page. That is a speed cost, not a visible one. |
| `pdf-graft.c`, `pdf-clean.c`, `pdf-clean-file.c`, `pdf-image-rewriter.c`, `pdf-recolor.c`, `pdf-shade-recolor.c`, `pdf-zugferd.c`, `pdf-subset.c` | `pdf_graft_*`, `pdf_clean_*`, `pdf_rewrite_images`, `pdf_recolor_*`, `pdf_subset_fonts` | none | O | Document rewriting, sanitising, recolouring, font subsetting and e-invoice (ZUGFeRD) extraction. These are authoring and post-processing tools (`mutool clean`/`recolor`), not reading, rendering or extraction. |
| `pdf-write.c` | `pdf_save_document`, `do_incremental`, `writeobject`, `writexref` | `write.rs` (`incremental_update`, `write_object`), `page_edit.rs`, `annot_edit.rs` | P | Only the incremental append with a classic xref table. **Missing:** full rewrite and garbage collection, xref-stream and object-stream output, linearized output, and encryption on write. Editing works (kopitiam appends). "Save as optimised/compressed" does not exist. |
| `pdf-label.c` | `pdf_label_object`, `pdf_load_object_labels` | none | O | This is MuPDF's *object* labeller, a debugging aid ("where is object 12 used"). Note that **page** labels (`/PageLabels`) live in `pdf-page.c`; see that row. |
| `pdf-af.c` | `pdf_count_document_associated_files`, `pdf_page_associated_file` | none | M | PDF 2.0 associated files (`/AF`) are not listed. Low impact: attached source files, and PDF/A-3 embedded XML, are invisible to the user. |

### PDF layer: pages, content interpretation, resources

| MuPDF source | Key functions | kopitiam-pdf target | Status | Consequence if not translated / notes |
|---|---|---|---|---|
| `pdf-page.c` | `pdf_load_page_tree`, `pdf_lookup_page_loc`, `pdf_flatten_inheritable_page_items`, `pdf_page_obj_transform_box`, `pdf_page_contents`, `pdf_page_label`, `pdf_load_links` | `xref.rs` (tree walk), `page_run.rs` (`page_ctm`, `gather_contents`), `page_geom.rs`, `link.rs` | P | **0.4.2 (2026-09-28):** **CropBox is the page** (`pdf_page_obj_transform_box` + `pdf_bound_page` + the CropBox clip; raster size, CTM, stext, page geometry), `UserUnit` applied. Still missing: page labels, separations/overprint query. _Audit text (snapshot, before 0.4.2):_ ~~The page tree, inheritance, `/Rotate` (snapped to 90°) and `/Contents` arrays (including an indirect array, fixed in 0.4.1) are ported. **Missing:** (1) **`CropBox`**: only the `MediaBox` is used (`page_run.rs:139-175`; MuPDF `pdf-page.c:765-809` intersects with the CropBox), so publisher and LaTeX pages show the bleed and crop marks, and text outside the crop area is extracted; (2) `UserUnit` is always 1, so oversized engineering drawings render at the wrong physical size; (3) **page labels** (`pdf_page_label`, `/PageLabels`), so the viewer shows "page 5" where the document says "iii"; (4) the page separations and overprint query.~~ |
| `pdf-interpret.c` | `pdf_process_stream`, `pdf_process_keyword`, `pdf_lookup_resource`, `pdf_process_annot`, `pdf_tos_*`, `parse_inline_image`, `pdf_process_BDC`/`EMC` (OC and ActualText gating) | `interpret.rs`, `resources.rs`, `annot_run.rs` | P | **0.4.2 (2026-09-28):** `gs`, `d`, `J`, `j`, `M`, `sh`, `d0`, `d1`, `BMC`/`BDC`/`EMC` (optional-content gating) are handled; inline images are parsed and decoded (`parse_inline_image`). ActualText is forwarded (inline, or from the MCID's structure element). `ri`/`i` stay ignored (no CMS; flatness is irrelevant). _Audit text (snapshot, before 0.4.2):_ ~~The tokenizer, dispatch and text-object state are faithful. **Stubbed (parsed and ignored):** `gs`, `d`, `J`, `j`, `M`, `ri`, `i`, `sh`, `d0`, `d1`, `MP`, `DP`, `BMC`, `BDC`, `EMC`, `BX`, `EX` (all of them fall into `_ => {}`, `interpret.rs:494-497`). **Missing:** decoding of inline images (`BI` is skipped by the byte scan at `interpret.rs:508-547`); optional-content culling (`pdf_is_ocg_hidden` inside `BDC /OC`, and on XObjects/annots); and ActualText. The consequences are in the feature table below.~~ |
| `pdf-op-run.c` | `pdf_run_*` operators, `pdf_show_char`, `pdf_show_image`, `pdf_show_pattern`, `pdf_show_shade`, `pdf_begin_group`, `pdf_run_xobject`, `pdf_flush_text` | `op_run.rs`, `page_run.rs` | P | **0.4.2 (2026-09-28):** ported since the audit: `pdf_show_pattern` (tiling, both branches), `pdf_show_shade` and shading patterns, clip paths (exact masks), text clip 4-7, stroke text 1/2/5/6, Type3 glyph execution, Form `/BBox` clip + `gbot`, `/OC` hiding, stencil and inline images. Transparency is ported too: `begin_softmask`/`end_softmask`, `pdf_begin_group`/`pdf_end_group` at every paint site, and the `pdf_run_xobject` group branch. Knockout is not; nor is overprint. _Audit text (snapshot, before 0.4.2):_ ~~38/153 functions named, the best-covered big file in the pdf layer. **Ported:** q/Q/cm, path construction and painting, `w`, colour operators, text showing (Tj/TJ/'/"), Form XObjects with `/Matrix` and a cycle guard, and Image XObjects. **Missing:** pattern and shading painting (`pdf_show_pattern`, `pdf_show_shade`), transparency groups and soft masks (`pdf_begin_group`, `begin_softmask`), clip paths (only a bbox approximation), text clip modes 4-7, stroke-text (`Tr 1/2` paint as fill), Type3 glyph execution, knockout, and overprint.~~ |
| `pdf-run.c` | `pdf_run_page_contents`, `pdf_run_page_annots`, `pdf_run_page_widgets`, `pdf_run_annot_with_usage` | `page_run.rs`, `annot_run.rs`, `draw_device.rs` `rasterize_page_ex` | P | Contents + annots are run for rendering. **Missing:** the `usage` ("View"/"Print") argument that drives OCG and annotation-flag selection (`Print`-only annots are drawn on screen too). Text extraction (`stext_device::page_to_stext`) runs **contents only**, so annotation and form text is missing from extraction (MuPDF's `fz_new_stext_page_from_page` includes it). |
| `pdf-resources.c` | `pdf_find_font_resource`, `pdf_insert_*_resource`, `pdf_purge_local_resources` | none | O | Despite the name, this file is the **write-side** resource de-duplication store used when authoring. The *lookup* (`pdf_lookup_resource`) is in `pdf-interpret.c` and is ported in `resources.rs`. `resources.rs:1` cites pdf-resources.c for lookup, which is misleading but harmless. |
| `pdf-xobject.c` | `pdf_xobject_matrix`, `pdf_xobject_bbox`, `pdf_xobject_isolated`, `pdf_xobject_knockout`, `pdf_xobject_transparency` | `page_run.rs` `op_do`/`matrix_from` | P | **0.4.2 (2026-09-28):** the `/BBox` clip and `/OC` are applied. The `/Group` attributes (transparency, isolated, knockout, colour space) are read too. Knockout is drawn non-knockout, and only a DeviceGray group `/CS` is honoured. _Audit text (snapshot, before 0.4.2):_ ~~`/Matrix` and `/Resources` are ported. **Missing:** the `/BBox` clip (a form's content spills outside its box), and the `/Group` transparency attributes (isolated, knockout, group colour space). Group soft masks are therefore ignored too.~~ |
| `pdf-layer.c` | `pdf_is_ocg_hidden`, `pdf_read_ocg`, `pdf_select_layer_config`, UI list | ~~none~~ `layer.rs` | ~~**M**~~ **P** | **0.4.2 (2026-09-28):** the default configuration is ported (`layer.rs`: `pdf_read_ocg` + `pdf_select_layer_config(-1)`, `pdf_is_ocg_hidden` incl. OCMD policies and Intent), applied to `BDC /OC` content and XObject `/OC`. Still missing: alternate configurations, the Print usage, a layer UI, and annotation `/OC`. _Audit text (snapshot, before 0.4.2):_ ~~No optional content. Layers that are OFF by default in `/OCProperties /D` (hidden watermarks, alternate languages, answer keys, CAD layers) **render and extract as if ON**. Also no layer UI.~~ |
| `pdf-struct.c` | `pdf_check_structure_tree` | none | M | Structure-tree validation. Low impact: MuPDF itself only consults structure for `FZ_STEXT_COLLECT_STRUCTURE`, which is not the default. |
| `pdf-util.c` | `pdf_new_pixmap_from_page_*`, `pdf_new_stext_page_from_annot` | `rasterize_page`, `page_to_stext` | O | Convenience wrappers. The two that matter have direct equivalents. The separations and usage variants are covered by other rows. |
| `pdf-device.c`, `pdf-op-buffer.c`, `pdf-op-filter.c`, `pdf-op-color.c`, `pdf-op-vectorize.c` | `pdf_new_pdf_device`, `pdf_new_buffer_processor`, `pdf_new_sanitize_filter`, `pdf_new_color_filter`, `pdf_new_vectorize_filter` | none | O | Processors that **write** or rewrite content streams (redaction, sanitising, recolouring, vectorising). Not needed to read. `annot_appearance.rs`/`form.rs` hand-write their few operator bytes instead. |

### PDF layer: fonts, CMaps, text encoding

| MuPDF source | Key functions | kopitiam-pdf target | Status | Consequence if not translated / notes |
|---|---|---|---|---|
| `pdf-font.c` | `pdf_load_font`, `pdf_load_simple_font`, `pdf_load_type0_font`, `load_cid_font`, `pdf_load_font_descriptor`, `pdf_load_embedded_font`, `pdf_load_substitute_font`, `select_truetype_cmap` | `font.rs`, `standard_font.rs`, `glyph*.rs` | P | **0.4.2 (2026-09-28):** Type3 `/FontMatrix` width scaling is fixed (item 3). The rest of this row stands. _Audit text (snapshot, before 0.4.2):_ ~~**Ported:** simple fonts, Type0/CID fonts, `/Widths`, `/W`+`/DW`, `/Encoding` + `/Differences`, `/ToUnicode`, `/CIDToGIDMap`, embedded FontFile/FontFile2/FontFile3 glyph selection, and base-14 substitution for non-embedded Latin fonts (0.4.1 added flag-based substitution for unknown Nonsymbolic fonts). **Missing:** (1) builtin/AFM widths when `/Widths` is absent (`standard_font.rs:55-62`), so a non-embedded base-14 font without `/Widths` lays out with every glyph at `/MissingWidth`, giving overlapping or stretched text and bad extraction spacing; (2) `/W2`/`/DW2` vertical metrics; (3) Type3 `/FontMatrix` width scaling (`font.rs:46-47`); (4) CJK and script fallback fonts for non-embedded fonts (MuPDF's `pdf_load_substitute_cjk_font` + Noto). The native render paints boxes there and hayro takes the page.~~ |
| `pdf-type3.c` | `pdf_load_type3_font`, `pdf_load_type3_glyphs`, `pdf_run_glyph` (+ `d0`/`d1` in op-run) | ~~none (`Type3` routed through `Font::load_simple`, `font.rs:214`)~~ `font.rs` (`Type3Info`), `op_run.rs` (`run_type3_glyph`) | ~~**M**~~ **T** | **0.4.2 (2026-09-28):** ported: `/FontMatrix`-scaled widths, FontBBox ascender/descender, glyph procedures run through the interpreter (`d0`/`d1`, colour masking for `d1`), no advance boxes and so no hayro fallback for Type3 text. _Audit text (snapshot, before 0.4.2):_ ~~Type3 glyph procs are never run. Native render: an **advance box per glyph**, which triggers the hayro whole-page fallback, so the screen is usually rescued. `rasterize_page_native` and any non-fallback caller show boxes. Extraction still gets Unicode from `/Encoding`/`/ToUnicode`, but advances are wrong unless FontMatrix = 0.001.~~ |
| `pdf-metrics.c` | `pdf_add_hmtx`, `pdf_end_hmtx`, `pdf_lookup_hmtx`, `pdf_lookup_vmtx` | `font.rs` (`Hmtx`) | T | Horizontal metrics are translated; vertical (`vmtx`) is not, as covered in the pdf-font row. |
| `pdf-cmap.c` | `pdf_add_cmap_range`, `pdf_lookup_cmap`, `pdf_lookup_cmap_full`, `pdf_decode_cmap`, `pdf_sort_cmap` | `cmap.rs` | T | The splay tree is replaced by a sorted `Vec` with the same lookup. Overlap splitting is not reproduced (`cmap.rs:23-31`), so a CMap that redefines an overlapping range keeps the first mapping where MuPDF keeps the last. That is rare. |
| `pdf-cmap-parse.c` | `pdf_parse_cmap`, `pdf_parse_bf_range`, `pdf_parse_cid_range` | `cmap.rs` | T | 11/13 functions named. |
| `pdf-cmap-load.c` (+ `source/pdf/cmaps/*.h`) | `pdf_load_system_cmap`, `pdf_load_builtin_cmap`, `pdf_load_embedded_cmap` | `cmap.rs` `load_predefined` (Identity only) | P | **0.4.2 (2026-09-28):** unchanged -- named CMaps are still not ported. The ledger row claiming them ported is corrected (the generator was reading a backticked path in a "Not ported" section). _Audit text (snapshot, before 0.4.2):_ ~~Embedded CMaps load. **Named predefined CMaps are not ported**; only `Identity-H`/`Identity-V` are (`cmap.rs:309-315`). Everything else falls back to Identity-H (`font.rs:367-371`). CJK documents using `UniGB-UCS2-H`, `90ms-RKSJ-H`, `UniJIS-UTF16-H` and so on get **misparsed codes**, which means garbled text, wrong advances and wrong glyphs. The Adobe `*-UCS2` collection maps (the ToUnicode fallback for CJK fonts with no `/ToUnicode`) are also missing, so such fonts extract as U+FFFD. `docs/port-ledger.md` marks `cmaps/*.h` as **ported**; see [Surprises](#surprises-and-defects-found-in-passing).~~ |
| `pdf-unicode.c` | `pdf_load_to_unicode`, `pdf_remap_cmap`, `pdf_new_identity_cmap` | `font.rs`, `cmap.rs` `remap` | T | 3/3 functions named. |
| `pdf-font-add.c` | `pdf_add_simple_font`, `pdf_add_cid_font`, `pdf_add_cjk_font` | none | O | Embedding fonts when **writing** (FreeText or form authoring with a real font). `form.rs` references base-14 by name instead. |

### PDF layer: images, colour, patterns, shadings, functions

| MuPDF source | Key functions | kopitiam-pdf target | Status | Consequence if not translated / notes |
|---|---|---|---|---|
| `pdf-image.c` | `pdf_load_image`, `pdf_load_image_imp`, `pdf_load_jpx`, `pdf_is_jpx_image` | `page_image.rs` | P | **0.4.2 (2026-09-28):** JPX (hayro-jpeg2000) and JBIG2 with `/JBIG2Globals` (hayro-jbig2), stencil `/ImageMask` painting, inline images, colour-key `/Mask` arrays and stencil `/Mask` streams are all ported. Still missing: `/Matte`; colour keys on DCT/JPX images. _Audit text (snapshot, before 0.4.2):_ ~~**Ported:** `/Width`/`/Height`/`/BPC`/`/Decode`/`/ImageMask` reading, 1/2/4/8/16-bpc, `/SMask` (added after the module header was written, see the doc-drift list), DCT, CCITT (0.4.0), and the Flate/LZW/AHx/A85/RL chains. **Missing:** JPX (`pdf_load_jpx`), JBIG2 with `/JBIG2Globals`, and colour-key and stencil `/Mask`. Image masks are treated as gray images rather than stencils. `/Interpolate` is ignored.~~ |
| `pdf-colorspace.c` | `pdf_load_colorspace`, `load_icc_based`, `load_indexed`, `load_devicen`, `load_separation`, `pdf_load_output_intent` | `resources.rs` `colorspace_from_obj` (fills), `page_image.rs` `parse_colorspace` (images) | P | **0.4.2 (2026-09-28):** Indexed, Separation/DeviceN (tint transforms via `function.rs`), Lab (`lab_to_rgb`), ICC-by-N and Pattern spaces are built as `pdf_load_colorspace` builds them, for fills and images alike, matching MuPDF's no-ICC (`-N`) conversions. Still missing: ICC management (see color-lcms), OutputIntent. _Audit text (snapshot, before 0.4.2):_ ~~Device spaces are exact. For **fills**: ICCBased is mapped by `/N`; CalGray/CalRGB are treated as Device (gamma ignored); **Lab is treated as RGB** (`resources.rs:219`), which gives wrong, saturated colours; Separation/DeviceN become a gray approximation with no tint transform (`resources.rs:75-85`); **Indexed passes the raw index as gray** (`resources.rs:86`), so nearly every indexed fill paints white. For **images**: Indexed is a proper palette lookup, but Separation/DeviceN/Lab error out and the image renders blank. The OutputIntent is ignored.~~ |
| `pdf-function.c` | `pdf_load_function`, `pdf_eval_function` (types 0/2/3/4 PostScript calculator) | ~~none~~ `function.rs` | ~~**M**~~ **T** | **0.4.2 (2026-09-28):** ported in full (`function.rs`: sampled, exponential, stitching, PostScript calculator; 25 unit tests). _Audit text (snapshot, before 0.4.2):_ ~~Needed for Separation/DeviceN tint transforms, shadings, transfer functions and soft-mask `/TR`. Without it, spot colours are guessed as gray and every shading is impossible. PANTONE-coloured logos, for example, render gray.~~ |
| `pdf-pattern.c` | `pdf_load_pattern`, `pdf_pattern_uses_blending` | ~~none~~ `op_run.rs` (`set_pattern`, `show_tiling_pattern`) | ~~**M**~~ **T** | **0.4.2 (2026-09-28):** ported: `pdf_load_pattern` (tiling), `pdf_pattern_uses_blending`, and the shading-pattern path. _Audit text (snapshot, before 0.4.2):_ ~~Tiling patterns (`/PatternType 1`) never paint. The pattern area is filled with the previous colour, or with nothing for `scn /P` without components (`op_run.rs:349-355`). Hatched map areas, textured backgrounds and some table shading come out wrong.~~ |
| `pdf-shade.c` | `pdf_load_shading`, `pdf_load_function_based_shading`, `pdf_load_axial_shading`, `pdf_load_radial_shading`, the mesh loaders (types 4-7) | ~~none~~ `shade.rs` | ~~**M**~~ **T** | **0.4.2 (2026-09-28):** ported: all seven shading types (`shade.rs`), code-to-code pixel-identical on the feature corpus. _Audit text (snapshot, before 0.4.2):_ ~~`sh` and shading patterns do not paint. **Every gradient is missing**, and whatever lies beneath shows.~~ |

### PDF layer: annotations, forms, links, outline, JS, signatures

| MuPDF source | Key functions | kopitiam-pdf target | Status | Consequence if not translated / notes |
|---|---|---|---|---|
| `pdf-annot.c` | `pdf_load_annots`, `pdf_annot_ap`, `pdf_annot_transform`, `pdf_create_annot`, `pdf_delete_annot`, `pdf_set_annot_*`, 195 functions total | `annot_run.rs`, `annot_edit.rs`, `annot_appearance.rs` | P | **Ported:** running `/AP /N` (and the `/AS` state dict) with Hidden/NoView/Popup skips, the Widget `/FT`+`/T` gate, and Ink annotation create/delete/edit. **Missing:** `NoRotate` on rotated pages (`annot_run.rs:27-34`); `/AP /D`/`/R` (fine for static rendering); the OC check on annots; the setters for every other subtype; and the whole redaction path. 17/195 named, but reading and rendering is the bulk of what a viewer needs. |
| `pdf-appearance.c` | `pdf_update_appearance`, `pdf_write_*_appearance` (Square, Circle, Line, Polygon, Highlight, Underline, StrikeOut, Squiggly, FreeText, Text icon, Stamp, Caret, FileAttachment, Widget Tx/Ch/Btn), `write_variable_text` | `annot_appearance.rs` (Ink only), `form.rs` (Tx widget regeneration, simplified) | P | An annotation **without `/AP`** renders only if it is Ink (`annot_appearance.rs:192-200`). Highlight, Square, FreeText, sticky-note icons and others that some producers write without `/AP` are **invisible**, where MuPDF synthesises them. Dash patterns in synthesised appearances are dropped on purpose (`annot_appearance.rs:26-36`). Text-field regeneration covers single and multi-line with a simple wrap, and approximates base-14 widths (`form.rs` module docs). |
| `pdf-form.c` | `pdf_field_value`, `pdf_set_field_value`, `pdf_toggle_widget`, `pdf_field_type`, `pdf_update_widget`, `pdf_choice_widget_*` | `form.rs`, `src/gui_frontend/forms.rs` | P | Text, checkbox and radio read/set/toggle work, and the appearance is regenerated. **Missing:** combobox/listbox value setting (classified, but not editable); signature fields (classified only); calculation order and format/keystroke events (these need JS, which is out of scope by design, `form.rs` module docs); and `/NeedAppearances` handling. |
| `pdf-layout.c` | `pdf_layout_fit_text`, `break_lines`, `measure_character` | none (`form.rs` has its own simpler wrap) | M | FreeText and widget auto-fit layout. Low impact: regenerated multi-line field text may wrap and size differently from MuPDF or Acrobat. |
| `pdf-link.c` | `pdf_load_link_annots`, `pdf_parse_link_dest`, `pdf_resolve_link`, `pdf_parse_link_action` | `link.rs`, `destination.rs` | P | GoTo, URI, named dests and explicit dests are ported, typed instead of MuPDF's URI-string round trip (a deliberate divergence, `destination.rs:21-29`). `/GoToR`, `/Launch` and `/Named` are recognised but not followed. Link *creation* is not ported. |
| `pdf-outline.c` | `pdf_load_outline`, `pdf_load_outline_imp` | `outline.rs` | T | Ported with a loop guard. The outline editing iterator is not ported (authoring). |
| `pdf-js.c`, `pdf-event.c` | `pdf_js_*`, `pdf_event_*` | none | O | The JavaScript engine glue (mujs). Out of scope by project decision (`form.rs`: "No JavaScript engine is required"). User-visible: calculated fields do not recalculate, and format masks (dates, currency) are not applied. |
| `pdf-signature.c` | `pdf_sign_signature`, `pdf_check_signature` hooks, `pdf_signature_*` | none | M | No signature verification or signing. A signed PDF still *renders* its signature appearance (via `/AP`), but no "signed by / valid" status is shown. Low impact for a reader. |

### fitz: devices and rasterisation (the draw path)

| MuPDF source | Key functions | kopitiam-pdf target | Status | Consequence if not translated / notes |
|---|---|---|---|---|
| `device.c` | `fz_fill_path`, `fz_stroke_path`, `fz_clip_*`, `fz_fill_text`, `fz_fill_image`, `fz_fill_image_mask`, `fz_fill_shade`, `fz_begin_group`, `fz_begin_mask`, `fz_begin_tile`, `fz_begin_metatext` | `text_device.rs` (`TextDevice` trait) | P | **0.4.2 (2026-09-28):** the trait grew clip path/text, pop clip, shade, tile, stroke glyph, image mask, filler chars, alpha. Groups and masks are there too (`begin_group`/`end_group`, `begin_mask`/`end_mask`), and so is ActualText (`begin_actualtext`/`end_actualtext`/`flush_text`). _Audit text (snapshot, before 0.4.2):_ ~~The device interface is collapsed to show_glyph, fill_path, stroke_path, draw_image, set_fill_color and set_text_render_mode. There are **no clip, mask, group, tile, shade or metatext calls**, so no device can implement those features until the trait grows them.~~ |
| `draw-device.c` | `fz_draw_fill_path`, `fz_draw_stroke_path`, `fz_draw_fill_text`, `fz_draw_fill_image`, `fz_draw_clip_*`, `fz_draw_begin_group`/`mask`/`tile`, `fz_draw_fill_shade` | `draw_device.rs` | P | **0.4.2 (2026-09-28):** clip masks (scissor + coverage mask), tiles (`fz_draw_begin_tile`/`end_tile`), shades, stencil masks, glyph alpha and the MuPDF image pipeline are ported. So are groups (with a real `group_alpha` for non-isolated groups) and soft masks (luminosity/alpha, `/BC`, `/TR`). Knockout is still missing. _Audit text (snapshot, before 0.4.2):_ ~~Fill, stroke, glyph and image are ported. Clip is a bbox intersection only. Groups, masks, tiles, shades and knockout are all missing (its own header says so, `draw_device.rs:47-51`). Glyphs always fill at alpha 1.0 (`draw_device.rs:380`).~~ |
| `draw-edge.c` | `fz_insert_gel`, `fz_scan_convert_aa`, `non_zero_winding_aa`, `even_odd_aa` | `draw_edge.rs` | P | A deliberate re-expression: a coverage-accumulating sweep with vscale 4 (MuPDF uses 15, `draw_edge.rs:52`). Nonzero and even-odd give the same result; anti-aliased edges are not bit-exact (AID-0052). |
| `draw-edgebuffer.c` | `fz_new_edgebuffer`, any-part-of-pixel scan conversion | none | M | MuPDF's alternative rasteriser. It is used for `fz_set_graphics_min_line_width` / "any part of a pixel" rules. Low impact: very thin rules can drop out where MuPDF keeps them, though the stroker's 0.35 px half-width floor mostly covers this (`draw_path.rs:256`). |
| `draw-rasterize.c` | `fz_set_aa_level`, `fz_set_text_aa_level`, `fz_set_graphics_min_line_width` | constants in `draw_edge.rs` | P | The AA level is fixed. There are no text-vs-graphics AA levels, no aliased mode, and no min-line-width setting. The user cannot choose "no anti-aliasing" or crisper text. |
| `draw-paint.c` | `fz_paint_span*`, `fz_paint_solid_color*`, `fz_paint_pixmap*` with alpha/mask variants | `draw_edge.rs` `composite`, ~~`draw_device.rs` `blend_rgb`~~ `draw_blend.rs` (paint_pixmap family) | P | **0.4.2 (2026-09-28):** painting through a per-pixel mask, onto an alpha destination, and `fz_paint_pixmap_with_bbox` (tiles) are ported. Still no N-channel/spot/overprint painters (RGB-only device). _Audit text (snapshot, before 0.4.2):_ ~~One RGB source-over painter. No N-channel, spot or overprint painters, and no painting through an alpha mask (which is why stencil masks and soft-mask groups cannot work).~~ |
| `draw-path.c` | `fz_flatten_fill_path`, `fz_flatten_stroke_path`, `fz_add_line_join`, `fz_add_line_cap`, `fz_dash_path`, `fz_flatten_dash_path` | `draw_path.rs` | P | **0.4.2 (2026-09-28):** `fz_dash_path` is ported (the dash walker) and the styled stroker is reachable from `J`/`j`/`M`/`d`/`gs`; the stroke width floor is MuPDF's 0.2 px. Still: anisotropic CTMs stroke with `max_expansion`. _Audit text (snapshot, before 0.4.2):_ ~~Fill flattening and a stroker with all join and cap styles are ported (`stroke_to_polygons_styled`), but the **interpreter never reaches the styled entry**. `stroke_to_polygons` hard-codes round/round (`draw_path.rs:227-235`), because `J`/`j`/`M` are not tracked. **`fz_dash_path` is not ported at all**, so dashed lines draw solid. Non-uniform CTMs stroke with `max_expansion` (a circular pen), so an anisotropically scaled stroke comes out too thick in one axis.~~ |
| `draw-affine.c` | `fz_paint_image`, `fz_paint_affine_*` (nearest, bilinear, colour/alpha variants), `fz_gridfit_matrix` | ~~`draw_device.rs` `draw_image_clipped`~~ `draw_affine.rs` | ~~P~~ **T** | **0.4.2 (2026-09-28):** ported for the RGB/gray device (`draw_affine.rs`): `fz_gridfit_matrix`, near/bilinear affine painters, the colour (stencil) painter, per-pixel mask, alpha destination. _Audit text (snapshot, before 0.4.2):_ ~~Nearest-neighbour inverse mapping only (`draw_device.rs:51`). No bilinear sampling and no gridfitting. Upscaled images look blocky, and image edges can leave hairline seams between tiled images.~~ |
| `draw-scale-simple.c` | `fz_scale_pixmap`, filter weights (`triangle`, `mitchell`, box) | ~~none~~ `draw_scale.rs` | ~~**M**~~ **T** | **0.4.2 (2026-09-28):** ported (`draw_scale.rs`): `fz_scale_pixmap` weights + `fz_subsample_pixmap`. WASH-1400's 228 CCITT pages are now pixel-identical to MuPDF. _Audit text (snapshot, before 0.4.2):_ ~~No down-scaling filter. A 300-dpi scan viewed at screen size is point-sampled, so thin strokes break up, 1-bit scans alias badly (moiré, dropped hairlines), and small text in scanned pages gets hard to read. MuPDF's "subsample + scale" path smooths these.~~ |
| `draw-unpack.c` | `fz_unpack_stream`, `fz_decode_tile` | `page_image.rs` `decode_samples`/`read_bits`/`component01` | T | 1-16 bpc unpacking and `/Decode` are ported. |
| `draw-blend.c` | `fz_blend_pixmap`, `fz_blend_*` (Multiply, Screen, … , Luminosity), knockout | ~~none~~ `draw_blend.rs` | ~~**M**~~ **T** | **0.4.2 (2026-09-28):** ported (`draw_blend.rs`): all 16 blend modes and `fz_blend_pixmap` with its isolated/non-isolated, separable/non-separable span blenders. Knockout (`fz_blend_pixmap_knockout`) is not ported. _Audit text (snapshot, before 0.4.2):_ ~~Every blend mode renders as Normal. Multiply-blended highlights and shadows paint as opaque blocks over the content.~~ |
| `draw-mesh.c` | `fz_paint_shade`, `fz_process_shade` | ~~none~~ `shade.rs` | ~~**M**~~ **T** | **0.4.2 (2026-09-28):** ported (`shade.rs`: Gouraud triangle painter, colour look-up). _Audit text (snapshot, before 0.4.2):_ ~~See shadings: gradients do not paint.~~ |
| `draw-glyph.c`, `glyph.c` | `fz_render_glyph`, glyph cache, `fz_subpixel_adjust`, `fz_render_stroked_glyph`, `fz_new_glyph_from_*` | none (every glyph is re-flattened as a path in `draw_device.rs` `show_glyph`) | M | **0.4.2 (2026-09-28):** stroked glyphs are drawn (`stroke_glyph`, `Tr 1/2/5/6`). The glyph cache is still missing (speed only). _Audit text (snapshot, before 0.4.2):_ ~~No glyph cache, so text-heavy pages re-flatten every outline on every render: slower page turns and zoom. No stroked-glyph rendering, so `Tr 1` outline text draws filled. No subpixel positioning cache. Visually close for fill text.~~ |
| `bbox-device.c`, `cull-device.c`, `list-device.c`, `test-device.c`, `trace-device.c`, `svg-device.c`, `xmltext-device.c`, `ocr-device.c` | the respective `fz_new_*_device` | none | O / M | `list-device.c` (display lists) is counted **missing**: the viewer re-interprets the page on every re-render instead of replaying a list. That costs speed, not correctness. The rest are out of scope: debugging and trace, SVG/XML output, OCR (kopitiam-ocr is a separate crate), and bbox/culling optimisations. |

### fitz: colour, pixmaps, images, shading

| MuPDF source | Key functions | kopitiam-pdf target | Status | Consequence if not translated / notes |
|---|---|---|---|---|
| `colorspace.c` | `fz_new_colorspace`, `fz_convert_color`, `fz_find_color_converter`, `fz_new_indexed_colorspace`, `fz_colorspace_n` | `resources.rs` `ColorSpace`, `page_image.rs` `ColorKind` | P | **0.4.2 (2026-09-28):** the enum now models Indexed, Separation/DeviceN, Lab and Pattern (`resources.rs` `ColorSpace`). Still no ICC. _Audit text (snapshot, before 0.4.2):_ ~~The colourspace object model is reduced to an enum that converts straight to DeviceRGB. No ICC, no `/Intent`, no black-point compensation, and no DeviceN/Separation model (see pdf-colorspace).~~ |
| `color-fast.c` | `gray_to_rgb`, `cmyk_to_rgb`, `lab_to_rgb`, … fast converters | `draw_device.rs` `gray_to_rgb`, `cmyk_to_rgb` | P | **0.4.2 (2026-09-28):** the CMYK formula is MuPDF's `1 - min(1, c + k)` and `lab_to_rgb` is ported. _Audit text (snapshot, before 0.4.2):_ ~~Only gray and CMYK to RGB. **The CMYK formula differs from MuPDF's**: the port uses `(1-c)(1-k)` (`draw_device.rs:481-487`), while MuPDF's no-ICC path is `1 - min(1, c+k)` (`color-fast.c:117-122`). Dark CMYK colours come out lighter than MuPDF's no-ICC result. `lab_to_rgb` is missing, which is why Lab is mis-mapped.~~ |
| `color-lcms.c`, `color-icc-create.c` | the LCMS2 ICC engine, CalRGB/CalGray profile synthesis | none | M | No colour management. ICC-tagged images and fills are treated as the device space with the same component count. Colours shift, most visibly on wide-gamut or CMYK press PDFs. MuPDF's default build is ICC-on, so this is a real rendering difference, not a pixel-exactness quibble. Not covered by AID-0052's substitution list, so strictly speaking it is an unrecorded omission. |
| `separation.c` | `fz_new_separations`, overprint simulation | none | M | No spot-colour separations and no overprint simulation. Overprinted black text over coloured backgrounds renders knocked-out, which is the correct *non*-overprint result. Print-preview-style overprint is not available. |
| `pixmap.c` | `fz_new_pixmap*`, `fz_clear_pixmap*`, `fz_convert_pixmap`, `fz_scale_pixmap`, `fz_subsample_pixmap`, `fz_invert_pixmap`, gamma, tinting | `pixmap.rs` (183 lines) | P | **0.4.2 (2026-09-28):** RGBA (premultiplied) pixmaps with an origin are used for tile and group layers. _Audit text (snapshot, before 0.4.2):_ ~~An RGB-only pixmap with new, clear and bbox. There is no alpha channel on the page pixmap, no N-component or spot pixmaps, and no gamma or tint.~~ |
| `image.c` | `fz_new_image_from_compressed_buffer`, `fz_get_pixmap_from_image`, `fz_decomp_image_from_stream`, image cache, subsampling by level | `page_image.rs` | P | **0.4.2 (2026-09-28):** subsampling by `l2factor` is in the draw pipeline (`draw_scale.rs`) and `fz_mask_color_key` is ported. Still no image cache. _Audit text (snapshot, before 0.4.2):_ ~~The decode pipeline is ported. There is no image cache (re-decoded per render) and no decode-at-reduced-resolution (`l2factor`), so large scans decode at full size every time: memory spikes and slow scrolling on big scans.~~ |
| `compressed-buffer.c` | `fz_open_image_decomp_stream` | `page_image.rs` `decode_image_base` | ~~P~~ **T** | **0.4.2 (2026-09-28):** the JPX and JBIG2 branches exist now (hayro-jpeg2000 / hayro-jbig2). _Audit text (snapshot, before 0.4.2):_ ~~The per-codec dispatch is ported for Flate/LZW/RL/AHx/A85/DCT/CCITT. JPX and JBIG2 branches are missing.~~ |
| `shade.c` | `fz_bound_shade`, `fz_process_shade` (function, linear, radial, mesh decomposition) | ~~none~~ `shade.rs` | ~~**M**~~ **T** | **0.4.2 (2026-09-28):** ported (`shade.rs`: `fz_process_shade` for all types, `fz_bound_shade`). _Audit text (snapshot, before 0.4.2):_ ~~See shadings.~~ |
| `halftone.c` | `fz_new_bitmap_from_pixmap`, threshold halftoning | none | O | Used only to produce 1-bit **output** (PCL/PBM). PDF `/HT` halftone dicts are ignored by MuPDF's screen renderer too, so there is no user-visible difference. |

### fitz: fonts and glyph outlines

| MuPDF source | Key functions | kopitiam-pdf target | Status | Consequence if not translated / notes |
|---|---|---|---|---|
| `font.c` | `fz_new_font_from_buffer`, `fz_outline_glyph`, `fz_outline_ft_glyph` (+ the `move_to`/`line_to`/`conic_to`/`cubic_to` decompose), `fz_bound_glyph`, `fz_encode_character`, `fz_advance_glyph` | `glyph.rs` (outline to Path callback shape, translated), `glyph_truetype.rs` (OpenType spec), `glyph_cff.rs` (Type2 spec + subset-cff container), `glyph_type1.rs` (Type1 spec), `glyph_skrifa.rs` (skrifa second opinion), `hayro_fallback.rs` (whole-page fallback) | D | FreeType is replaced per AID-0052, with skrifa added later under the Pure-Rust-Core "prefer a crate" rule. **Residual gaps:** there are no glyph bounding boxes from the font (so no `ACCURATE_BBOXES` for stext); no advance widths from the font program (see pdf-font: AFM/`hmtx` fallback); no hinting (the same as MuPDF's default, unhinted); and the documented ceilings (predefined-Expert CFF, some `seac` cases; GH #67) paint boxes, which hand the page to hayro. |
| `noto.c` | `fz_lookup_builtin_font`, `fz_lookup_cjk_font`, `fz_lookup_noto_font` | `standard_font.rs` (base-14 faces taken from hayro's bundled fonts) | P | Base-14 substitution exists. **No CJK or script (Noto) fallback faces.** A non-embedded Chinese, Japanese, Korean, Arabic, Thai or other font paints boxes natively (hayro rescues the screen; `rasterize_page_native` does not). |
| `subset-cff.c` | `subr_bias`, the INDEX/DICT/charset/FDSelect readers, `execute_charstring` | `glyph_cff.rs` (container parse adapted, per AID-0052's recorded exception) | P | The reading half is adapted. The subsetting half is write-side and out of scope. |
| `subset-ttf.c`, `harfbuzz.c`, `text-decoder.c`, `hyphen.c` | TTF subsetting, HarfBuzz shaping, text encoders, hyphenation | none | O | Authoring and HTML-layout support. PDF text is already positioned glyphs and needs no shaping. |
| `text.c` | `fz_new_text`, `fz_show_glyph`, `fz_bound_text`, `fz_text_language` | `text_device.rs` (collapsed per-glyph seam) | P | **0.4.2 (2026-09-28):** clip-text and stroke-text are expressible now (`clip_text`, `stroke_glyph`). _Audit text (snapshot, before 0.4.2):_ ~~`fz_text` span buffering is replaced by direct per-glyph device calls. Fine for filling. It is why clip-text and stroke-text cannot be expressed, and there is no language tag (MuPDF uses `/Lang` for stext).~~ |
| `encodings.c` | `fz_unicode_from_glyph_name`, `fz_glyph_name_from_unicode_sc`, the base-encoding tables | `encodings.rs`, `agl.rs`, `agl_data.rs`, `standard_encodings.rs` | T | Tables entry-for-entry. |
| `ucdn.c` | `ucdn_get_general_category`, `ucdn_compat_decompose`, `ucdn_get_bidi_class`, `ucdn_mirror` | `unicode-general-category`, `unicode-normalization` crates (Cargo.toml note) | D | Substituted. Bidi class and mirroring are not needed until bidi exists. |
| `bidi.c`, `bidi-std.c` | `fz_bidi_fragment_text`, `fz_bidi_resolve_*` | none | **M** | stext has no bidi (`stext_device.rs:50-52`). Arabic and Hebrew extract in visual order, reversed per line; copy-paste and search on RTL text break. |

### fitz: structured text (extraction)

| MuPDF source | Key functions | kopitiam-pdf target | Status | Consequence if not translated / notes |
|---|---|---|---|---|
| `stext-device.c` | `fz_new_stext_device`, `fz_add_stext_char_imp`, `add_char_to_line`, `fixup_bboxes_and_bidi`, `fz_stext_begin_metatext` (ActualText), `fz_stext_fill_image`, `fz_stext_clip_*`, style collection | `stext_device.rs`, `structured_text.rs` | P | **0.4.2 (2026-09-28):** `FZ_STEXT_CLIP` culling and ActualText (`do_extract_within_actualtext`, `flush_actualtext`, begin/end metatext with content-bounds placement) are ported. ACCURATE_BBOXES, styles, images, wmode and bidi are still missing. _Audit text (snapshot, before 0.4.2):_ ~~The char/line/block assembly, space synthesis, fake-bold drop, ligature decomposition and combining marks (0.4.2) are ported, and the core bug fixes (spurious spaces) are faithful. **Missing:** ActualText (`FZ_STEXT_IGNORE_ACTUALTEXT` is a no-op because nothing replaces anything); `FZ_STEXT_CLIP` (text clipped away, or hidden behind a clip, is still extracted); `ACCURATE_BBOXES` (char quads come from scalar ascender/descender, so highlight boxes are a bit loose); `COLLECT_STYLES`, `COLLECT_VECTORS` and `PRESERVE_IMAGES` (no image blocks, because the stext device's `draw_image` is the default no-op); vertical `wmode` (stubbed); and bidi.~~ |
| `stext-boxer.c` | `fz_segment_stext_page`, `boxer_*` whitespace-cover segmentation | `stext_boxer.rs` | T | 19/29 functions named, and the algorithm is ported. This is what fixes two-column interleave. |
| `stext-para.c` | `fz_paragraph_break`, line-walker heuristics (indent, bullets, justify) | `stext_para.rs` | P | Only the line-gap breaker (`stext_para.rs` header). Paragraphs split on vertical gap only. Indent, bullet and justification cues are not used, so paragraphs in single-spaced documents merge or split differently from MuPDF. |
| `stext-classify.c` | `fz_classify_stext_rect` | `stext_classify.rs` | P | There is no struct-block tree to split, so classification assigns a line to the region it overlaps most. Faithful splitting is deferred (header). |
| `stext-iterator.c` | `fz_stext_page_block_iterator_*` | `stext_iterator.rs` | T | Depth-first order is equivalent. |
| `stext-search.c` | `fz_search_stext_page`, `canon`, `hdist`/`vdist`, `fz_highlight_selection`, `fz_copy_selection`, `fz_snap_selection`, `fz_copy_rectangle` | `stext_search.rs` | P | Search is ported, with two documented deliberate divergences (a line break counts as a space, and whitespace runs collapse). **Selection APIs are not ported**: `fz_highlight_selection`, `fz_copy_selection`, `fz_snap_selection` and `fz_copy_rectangle` are not found anywhere in `src/`. Drag-to-select, copy text of a selection, and word/line snapping need a separate implementation or port. |
| `stext-table.c` | `fz_table_hunt`, grid finding | none | M | No table detection (`FZ_STEXT_TABLE_HUNT` is a no-op). Tables extract as plain lines, with no cell structure. |
| `stext-raft.c` | the raft (line-fragment grouping) helpers | none | M | Low impact: used by MuPDF's newer segmentation and table paths, which are not ported either. |
| `stext-output.c` | `fz_print_stext_page_as_text`/`html`/`xhtml`/`json`/`xml` | `src/mupdf_extract.rs` (`stext_to_page`: line to `TextSpan`) | O | Output formatting. kopitiam converts stext into its own `Page` model and renders downstream. `mutool draw -F html/json`-style output does not exist, by design. |

### fitz: filters, streams, crypto primitives

| MuPDF source | Key functions | kopitiam-pdf target | Status | Consequence if not translated / notes |
|---|---|---|---|---|
| `filter-basic.c` | `fz_open_ahxd`, `fz_open_a85d`, `fz_open_rld`, `fz_open_null_filter`, `fz_open_endstream_filter`, `fz_open_range_filter`, `fz_open_concat`, `fz_open_arc4`, `fz_open_aesd` | `filter_basic.rs`, `crypt.rs` (arc4/aesd via crates) | T | **0.4.2 (2026-09-28):** `fz_open_endstream_filter` semantics are ported now (xref.rs). _Audit text (snapshot, before 0.4.2):_ ~~The decode filters are faithful. **But `fz_open_endstream_filter` (the bad-`/Length` recovery) is not ported.** Its consequence is in the pdf-stream row. `concat` is replaced by byte concatenation in `page_run.rs` `gather_contents`.~~ |
| `filter-flate.c` | `fz_open_flated`, `next_flated` | `filter_flate.rs` (wiring) + `miniz_oxide` | D | AID-0052. MuPDF recovers some truncated zlib streams; the port's behaviour on corrupt Flate is miniz_oxide's. |
| `filter-lzw.c` | `fz_open_lzwd`, `next_lzwd` | `filter_lzw.rs` | T | EarlyChange is ported. |
| `filter-predict.c` | `fz_open_predict`, PNG/TIFF predictors | `filter_predict.rs` | T | 7/8 functions named. |
| `filter-fax.c` | `fz_open_faxd`, `dec1d`, `dec2d` | `filter_fax.rs`, `filter_fax_tables.rs` | T | Landed in 0.4.0. |
| `filter-dct.c`, `load-jpeg.c`, `jmemcust.c` | `fz_open_dctd`, `fz_load_jpeg` | `page_image.rs` + `zune-jpeg` | D | AID-0052. The Adobe APP14 inverted-CMYK convention is handled via `/Decode` (`page_image.rs:514`). DCT is decoded only on the image path. A DCT-filtered *non-image* stream (which is legal but rare) passes through undecoded (`doc_stream.rs:134`). |
| `filter-jbig2.c` (+ jbig2dec) | `fz_open_jbig2d`, `fz_load_jbig2_globals` | ~~none (`page_image.rs:140-141` rejects)~~ `page_image.rs` + `hayro-jbig2` | ~~**M**~~ **D** | **0.4.2 (2026-09-28):** substituted by `hayro-jbig2` (=0.3.0, pure Rust), with `/JBIG2Globals` and MuPDF's polarity inversion. _Audit text (snapshot, before 0.4.2):_ ~~**High impact.** JBIG2-compressed scans render blank and cannot be OCR'd. Not covered by AID-0052. A pure-Rust JBIG2 decoder (e.g. hayro's own `hayro-jbig2`) would be a Pure-Rust-Core-compliant substitute to evaluate.~~ |
| `load-jpx.c` (+ OpenJPEG) | `fz_load_jpx`, `jpx_read_image` | ~~none (`page_image.rs:140-141` rejects)~~ `page_image.rs` + `hayro-jpeg2000` | ~~**M**~~ **D** | **0.4.2 (2026-09-28):** substituted by `hayro-jpeg2000` (=0.3.5, pure Rust) with MuPDF's colour-space rule; 211 NUREG/CR-7289 figures that rendered blank now match MuPDF. _Audit text (snapshot, before 0.4.2):_ ~~JPEG 2000 images render blank. The same kind of substitute question applies (e.g. `hayro-jpeg2000`). Neither crate's licence, pure-Rust build or API was checked here: to be verified before anyone relies on this.~~ |
| `filter-brotli.c` | `fz_open_brotlid` | none | M | MuPDF's non-standard `BrotliDecode` extension. Very rare in the wild. Such a stream is returned raw (`doc_stream.rs` passes unknown filters through), so the content comes out garbage. |
| `filter-sgi.c`, `filter-thunder.c`, `filter-leech.c` | SGI LogLuv / Thunderscan (TIFF-only), leech (stream capture) | none | O | Not PDF filters. |
| `stream-open.c`, `stream-read.c` | `fz_open_memory`, `fz_read`, `fz_seek`, `fz_read_best`, `fz_read_byte` | `stream.rs` | T | The file-backed stream is replaced by in-memory (the whole file is loaded). Fine for local files, but a very large PDF costs its full size in RAM. |
| `crypt-aes.c`, `crypt-arc4.c`, `crypt-md5.c` | AES-CBC, RC4, MD5 | `aes`/`cbc`, `rc4`, `md-5` crates | D | RustCrypto substitutes (Cargo.toml notes). |
| `crypt-sha2.c` | SHA-256/384/512 | ~~none~~ `crypt.rs` + `sha2` | ~~M~~ **D** | **0.4.2 (2026-09-28):** substituted by the `sha2` crate (0.10), used by the R5/R6 key derivation. _Audit text (snapshot, before 0.4.2):_ ~~Only needed for R5/R6 (AES-256) key derivation, which is not implemented. See pdf-crypt.~~ |
| `encode-basic.c`, `encode-fax.c`, `encode-jpx.c`, `compress.c`, `brotli.c`, `output*.c`, `bitmap.c`, `writer.c` | encoders and document writers | none | O | Output encoders and writers (PNG/PNM/PS/PCL/PWG/SVG/DOCX/CBZ/CSV, fax/JPX encoding). The viewer produces pixmaps, not files. |

### fitz: runtime, containers, utilities

| MuPDF source | Key functions | kopitiam-pdf target | Status | Consequence if not translated / notes |
|---|---|---|---|---|
| `geometry.c` | `fz_concat`, `fz_transform_rect`, `fz_intersect_rect`, … | `geometry.rs` | T | 50/73 named. |
| `path.c` | `fz_moveto`, `fz_lineto`, `fz_curveto`, `fz_closepath`, `fz_rectto`, `fz_walk_path`, `fz_bound_path`, `fz_new_stroke_state_with_dash_len` | `draw_path.rs` | P | **0.4.2 (2026-09-28):** a stroke state exists now (`draw_path::StrokeStyle`: width, caps, join, miter, dash). _Audit text (snapshot, before 0.4.2):_ ~~Path construction is faithful. There is no `fz_stroke_state` (dash array, caps and joins are not stored per path), which is the storage half of the dash and line-style gap.~~ |
| `buffer.c`, `string.c`, `hash.c`, `pool.c`, `error.c` | buffers, UTF-8 runes, hash table, bump pool, error taxonomy | `buffer.rs`, `string_util.rs`, `hash.rs`, `pool.rs`, `error.rs` | T | The foundation layer. `setjmp`/`longjmp` is replaced by `Result` (AID-0051). |
| `context.c`, `memory.c`, `store.c`, `list.c`, `tree.c`, `heap.c`, `ftoa.c`, `strtof.c`, `printf.c` | `fz_new_context`, allocators, resource store, containers, number formatting | Rust ownership + std (`Vec`, `HashMap`, `format!`, `str::parse`) | D | AID-0051 idiom mapping. There is no global store, so no memory-budgeted eviction (`fz_store_scavenge`). |
| `link.c`, `outline.c` | `fz_link`, `fz_outline` (+ outline iterator) | `link.rs`, `destination.rs`, `outline.rs` | T | Typed Rust structs. The outline editing iterator is not needed for reading. |
| `util.c` | `fz_new_pixmap_from_page`, `fz_new_stext_page_from_page`, `fz_search_page` | `rasterize_page`, `page_to_stext`, `stext_search` | P | The wrappers that matter exist. The display-list, separation and buffer-from-page variants do not, with no user-visible effect beyond the rows above. |
| `document.c`, `document-all.c`, `gz-doc.c` | the `fz_document` multi-format handler registry | none (callers use `PdfDocument` directly) | O | kopitiam handles only PDF through this path. XPS/EPUB/CBZ/HTML/SVG/image documents are separate or out of scope. |
| `archive.c`, `zip.c`, `unzip.c`, `untar.c`, `unlibarchive.c`, `uncfb.c`, `directory.c` | archive readers | none | O | Containers for EPUB, CBZ and XPS, not for PDF. |
| `load-bmp/gif/png/pnm/psd/tiff/jxr/jbig2.c` | standalone image-file loaders | none | O | Opening image *files* as documents. The in-PDF codecs are the filter rows above. |
| `transition.c`, `getopt.c`, `json.c`, `options.c`, `xml.c`, `xml-write.c`, `random.c`, `regexp.c`, `time.c`, `log.c`, `memento.c`, `track-usage.c`, `barcode.c`, `leptonica-wrap.c`, `deskew.c`, `skew.c`, `warp.c`, `glyphbox.c` | slideshow transitions, CLI, JSON/XML, RNG, debug allocators, barcode, OCR preprocessing, deskew/dewarp, glyph-outside-box test | none | O | None of these read, render or extract PDF input. `glyphbox.c`'s `fz_glyph_entirely_outside_box` belongs to the stext `FZ_STEXT_CLIP` feature and would come in with it. OCR preprocessing lives in `kopitiam-ocr`. |

## Feature-level breakdown: content-stream operators and render features

(0.4.2: the struck-through status is the snapshot's; the bold one is now. The
"What the user sees" column describes the snapshot.)

The file table rolls these up. This table is the per-feature view, and it is where `stubbed` shows up. Verified against `interpret.rs` `process_keyword` (lines 371-501).

| Feature | MuPDF home | Port status | What the user sees |
|---|---|---|---|
| `q Q cm` | pdf-op-run.c | translated | – |
| `m l c v y h re`; `S s f F f* B B* b b* n` | pdf-op-run.c, draw-path.c | translated | – |
| `W W*` clip | pdf-op-run.c `pdf_run_W`, draw-device.c `fz_draw_clip_path` | ~~**partial** (bbox only, `op_run.rs:297-313`)~~ **translated (exact masks, 0.4.2)** | Non-rectangular clips over-paint up to their bounding box. |
| `w` | pdf-op-run.c | translated | – |
| `d` dash | pdf-op-run.c, draw-path.c `fz_dash_path` | ~~**stubbed** (operands consumed, ignored)~~ **translated (0.4.2)** | Dashed and dotted lines draw solid. |
| `J j M` | pdf-op-run.c | ~~**stubbed**~~ **translated (0.4.2)** | Every stroke has round caps and joins. Square-ended rules look slightly longer, and mitred corners look rounded. |
| `ri i` | pdf-op-run.c | stubbed | None (intent needs ICC; flatness is irrelevant at screen resolution). |
| `gs` ExtGState | pdf-op-run.c `pdf_run_gs` + `pdf_run_extgstate` | ~~**stubbed**~~ translated (0.4.2: CA/ca, LW/LC/LJ/ML/D/Font, BM, SMask incl. /BC /TR; OP not simulated) | No alpha, blend mode, soft mask, `LW`/`LC`/`LJ`/`ML`/`D`/`Font`/`TR`/`OP` via gs. **Translucent content paints opaque.** |
| `g G rg RG k K` | pdf-op-run.c | ~~translated (CMYK formula differs from MuPDF's no-ICC path)~~ **translated (0.4.2: MuPDF's no-ICC CMYK formula)** | Dark CMYK comes out slightly lighter than MuPDF. |
| `cs CS sc SC scn SCN` | pdf-op-run.c, pdf-colorspace.c | ~~**partial**~~ **translated for the no-ICC path (0.4.2)** | See the colour-space gap (Indexed white, Lab red, spot gray, pattern = previous colour). |
| `sh` | pdf-op-run.c `pdf_run_sh`, pdf-shade.c | ~~**stubbed**~~ **translated (0.4.2)** | Gradients missing. |
| Shading and tiling `/Pattern` fills | pdf-op-run.c `pdf_show_pattern`, pdf-pattern.c | ~~**missing**~~ **translated (0.4.2)** | Patterned areas paint in the previous colour, or not at all. |
| `BT ET Tc Tw Tz TL Tf Tr Ts Td TD Tm T* Tj TJ ' "` | pdf-interpret.c, pdf-op-run.c | translated | – |
| `Tr 0` fill / `3` invisible | pdf-op-run.c `pdf_flush_text_imp` | translated (0.4.1) | – |
| `Tr 1 2` stroke text | same | ~~**partial** (paints as fill)~~ **translated (0.4.2)** | Outline-only headings draw as solid letters. |
| `Tr 4-7` clip text | same | ~~**partial** (4-6 paint, 7 skipped, no clip)~~ **translated (0.4.2)** | Text-as-clip-mask effects (an image seen through the letters) draw as plain text with the image unclipped. |
| `Tf` + embedded fonts | pdf-font.c, font.c | substituted-by-design | Real outlines for TrueType/CFF/Type1. Boxes plus hayro fallback at the documented ceilings. |
| Type3 `d0 d1` + CharProcs | pdf-type3.c | ~~**stubbed** (`d0`/`d1`) / missing (procs)~~ **translated (0.4.2)** | Boxes natively; hayro renders the page. |
| `Do` Form XObject | pdf-op-run.c `pdf_run_xobject` | ~~partial (no `/BBox` clip, no `/Group`, no `/OC`)~~ translated (0.4.2: /BBox clip, /Group transparency, /OC; knockout drawn non-knockout) | Forms spill outside their box. Transparency groups flatten. Hidden-layer forms show. |
| `Do` Image XObject | pdf-op-run.c `pdf_show_image` | ~~partial~~ **translated (0.4.2: JPX/JBIG2, all colour spaces, stencils, `/Mask`, MuPDF scaling; `/Matte` missing)** | JPX/JBIG2/Separation/DeviceN/Lab images are blank. Image masks paint black on white. `/Mask` is ignored. `/SMask` works. Nearest-neighbour sampling. |
| `BI ID EI` inline image | pdf-interpret.c `parse_inline_image`, pdf-op-run.c | ~~**stubbed** (skipped by byte scan)~~ **translated (0.4.2)** | Inline images blank. |
| `MP DP BMC BDC EMC` | pdf-interpret.c (OC gating), pdf-op-run.c `begin_metatext` | ~~**stubbed**~~ **partial (0.4.2: optional content gated; ActualText applied, inline and via the MCID's structure element)** | Hidden OC content shows. ActualText is not applied to extraction. Tagged structure is not collected. |
| `BX EX` | pdf-interpret.c | stubbed (correctly: the compatibility section just suppresses errors) | – |
| Knockout and isolated groups, soft-mask groups, blend modes | draw-device.c, draw-blend.c | ~~missing~~ translated (0.4.2) except knockout | Transparency-heavy designs (Illustrator and InDesign exports) render wrong. |
| Overprint and spot separations | separation.c, draw-paint.c | missing | No overprint simulation (the same as MuPDF's default screen render unless enabled). |
| Glyph cache | draw-glyph.c | missing | Speed only. |
| Anti-aliasing levels | draw-rasterize.c | partial (fixed) | Not user-configurable. |
| Image down-scaling and smoothing (`draw-scale-simple.c`), 1-bit smoothing | draw-scale-simple.c, draw-affine.c | ~~missing~~ **translated (0.4.2)** | Scans alias and lose thin strokes when zoomed out. |
| Transfer functions (`/TR`, `/TR2`) and halftones (`/HT`) | pdf-op-run.c (TR via functions), halftone.c | ~~missing / out-of-scope~~ soft-mask /TR translated (0.4.2); /TR /TR2 in ExtGState ignored as by MuPDF's screen path; /HT out of scope | TR rarely matters on screen. HT is ignored by MuPDF's screen path too. |

## Doc claims the code contradicts

These are **reported, not fixed**. The orchestrator owns the fixes. Each one was checked against the code quoted.

**0.4.2 (2026-09-28): all of them are now corrected in place**, struck through with a dated `CORRECTED` note (commits `61473db` and the tranche commits), except where the code itself changed underneath the claim (`draw_device.rs`, `resources.rs`, `interpret.rs`, `op_run.rs` were rewritten by the tranches and their headers along with them).

| File:line | The claim | What the code actually does |
|---|---|---|
| `crates/kopitiam-pdf/src/mupdf/mod.rs:61-65` | "Still ahead: … the xref / document layer …, encryption), the content interpreter, fonts/CMaps/ToUnicode, and the `stext` device + layout analysis … Not built yet, hor." | All of these exist: `xref.rs`, `crypt.rs`, `interpret.rs`, `font.rs`, `cmap.rs`, `stext_device.rs`, `stext_boxer.rs`, `stext_para.rs`. The module header is also still framed as "text-extraction vertical" and does not mention the draw path. |
| `crates/kopitiam-pdf/src/mupdf/xref.rs:38-40` | Deferred: "… encryption (`pdf-crypt.c`) …" | Encryption is implemented: `decryptor` (`xref.rs:98`), `build_decryptor` (:200), `rewrite_decrypted` (:308), backed by `crypt.rs`. |
| `crates/kopitiam-pdf/src/mupdf/font.rs:19-30` and `:44-49` | "this port **avoids FreeType entirely** … never rasterises a glyph"; deferred: "embedded-font glyph reading (FreeType)" | `font.rs` loads embedded programs (`load_font_program`, `CffProgram`, `Type1Program`, the `glyph_outline` used by `draw_device.rs:370`), and `glyph_truetype.rs`/`glyph_cff.rs`/`glyph_type1.rs`/`glyph_skrifa.rs` produce outlines. "No FreeType" is still true; "never reads glyphs" is not. |
| `crates/kopitiam-pdf/src/mupdf/draw_device.rs:39-41` | "**Invisible text.** The glyph sink has no render-mode, so `Tr 3/7` … still paints a box." | The render mode *is* plumbed through (`text_render_mode` field, `draw_device.rs:89-92`; `set_text_render_mode`, :447), and modes 3/7 return early (:356-358). |
| `crates/kopitiam-pdf/src/mupdf/op_run.rs:28-33` | Deferred: "actual stroking/filling/clipping … not ported (no rasterisation on the text path)" | `op_run.rs` has `op_paint`, `fill_current`, `stroke_current` and the pending-clip logic (`:262-313`). |
| `crates/kopitiam-pdf/src/mupdf/interpret.rs:494-496` | Comment in the `_ =>` arm: "Everything else (paths, colours, clips, shadings, gs, marked content, Type3 metrics, BX/EX) is parsed-and-ignored" | Paths, colours and clips have explicit arms above it (:450-486). Only shadings, gs, marked content, Type3 metrics, BX/EX, `d`, `J`, `j`, `M`, `ri` and `i` fall through. |
| `crates/kopitiam-pdf/src/mupdf/page_image.rs:43-45` | "Soft masks / `/SMask` / stencil-mask compositing are not applied -- each image is returned as its own opaque sample buffer." | `/SMask` **is** decoded (`decode_smask`, :279) and composited per pixel (`draw_device.rs:302-318`). The stencil (`/ImageMask`) and `/Mask` parts of the claim are still true. |
| `crates/kopitiam-pdf/src/mupdf/resources.rs:51-53` | `Indexed`: "approximated (TODO) as a gray ramp of `index / hival`" | `to_rgb` returns `[c(0), c(0), c(0)]` (:86), i.e. the **raw index**, not divided by `hival`. Any index ≥ 1 clamps to white in `rgb_to_bytes`. |
| `crates/kopitiam-pdf/src/mupdf/resources.rs:46` | `IccN`: "An N-component ICCBased/Lab space we approximate purely by component count." | `Lab` never reaches `IccN`. `colorspace_from_obj` maps `b"CalRGB" \| b"Lab" => ColorSpace::Rgb` (:219), so Lab components are used as RGB. |
| `crates/kopitiam-pdf/src/mupdf/draw_device.rs:478-479` | "the naive conversion fz_cmyk_to_rgb uses when no ICC profile applies" | MuPDF's no-ICC `cmyk_to_rgb` is `1 - min(1, c + k)` (`vendor/…/color-fast.c:117-122`); the port computes `(1-c)(1-k)` (:481-487). The breadcrumb misattributes the formula. |
| `crates/kopitiam-pdf/src/mupdf/structured_text.rs:38-40` | "the interpreter is not on the image path yet, so no image blocks are produced" | The interpreter *is* on the image path (`op_do` → `draw_image_xobject` → `TextDevice::draw_image`). Image blocks are absent because `StextDevice` keeps the trait's default no-op `draw_image` (`text_device.rs:114`). The outcome is the same; the stated reason is stale. |
| `crates/kopitiam-pdf/src/mupdf/stext_device.rs:59-60` | "Layout analysis (reading order / paragraphs) is the *next* wave." | Already there: `stext_boxer.rs`, `stext_para.rs` and `stext_classify.rs` exist, and `page_to_stext_segmented` is the extraction entry (`mupdf_extract.rs:59`). |
| `crates/kopitiam-pdf/src/mupdf/write.rs:66` | "(we have no `/Encrypt` support at all …)" | Reading encrypted files is supported (`crypt.rs`). The accurate statement is "no encryption on write; encrypted inputs are rewritten as plaintext at open". |
| `docs/port-ledger.md:49` (machine-generated) | `cmaps/*.h` → `cmap.rs`, status **ported** | `cmap.rs` ports *no* named CMap tables; `load_predefined` knows only Identity-H/V (`cmap.rs:309-315`), and its own header says so ("Not ported (deliberately)", :34-42). The ledger generator is reading a provenance header, not the code, so the fix belongs in the header the generator scrapes. |

## Surprises and defects found in passing

Not fixed, just reported. (**0.4.2:** 1, 2 and 3 are fixed -- stream decryption uses the real generation (tranche 5, with a MuPDF-made fixture), `gs` is honoured so the Ink opacity reaches the pixels (tranche 3), and the styled stroker is reachable (tranche 3). 4 stands as a design question. 5: `page_images` still fails the whole page on an undecodable image, but JPX/JBIG2 no longer are.)

1. **Encrypted streams are always decrypted with generation 0.** `xref.rs:807`: `d.decrypt_stream(num as u32, 0, &raw)`. Strings use the real generation (`xref.rs:604`), but streams hard-code 0. The per-object key mixes in the generation (§7.6.3.3 Algorithm 1), so any encrypted **stream** whose object has generation ≠ 0 (typical after an incremental update that reused a freed object number) decrypts to garbage. Its page content or image then comes out blank or corrupt. Because `open` rewrites the document to plaintext through this path, the damage is baked into the in-memory copy.
2. **The synthesised Ink appearance's opacity is silently dropped.** `annot_appearance.rs` builds an `/ExtGState` with `CA`/`ca` and emits `/GS0 gs`, and a unit test asserts it is there (:664-668). But the interpreter ignores `gs`, so the opacity never reaches the pixels. The test proves the bytes are written, not that the render honours them.
3. **The `J`/`j`/`M` stroker is ported but unreachable from content streams.** `stroke_to_polygons_styled` is only called from inside `draw_path.rs` (`grep -rln stroke_to_polygons_styled` gives just that file). `annot_appearance.rs` writes `1 J 1 j` "faithfully", and it happens to match only because the default is hard-coded round/round.
4. **The hayro fallback's scope is narrower than its doc suggests for a reader.** It triggers only on box glyphs (`draw_device.rs:391`). A page with a blank JBIG2 scan, missing gradients or opaque "transparent" boxes has zero box glyphs, so it is never re-rendered, even though hayro would draw it correctly. It is worth deciding whether "an unsupported image codec or shading was skipped" should also count toward the fallback.
5. **`page_images` fails the whole page when any image uses a deferred codec** (`page_image.rs:159-165`). In the draw path it is per image (`page_run.rs:286-289`), but the OCR helper path is all-or-nothing.

## Code-to-code harness results

Full methodology, per-tranche tables and the metric amendments are in
[`mupdf-code-to-code.md`](mupdf-code-to-code.md). The summary below is the
final measurement: `mutool` 19f1284 as the oracle, run with `-N -M 0` and the
per-channel gross metric. Both kopitiam-pdf 0.4.1 and 0.4.2 were measured
with that same final harness on 2026-09-28. 0.4.1 was built from the
harness commit `9955131`, whose library is 0.4.1's.

**Open corpus** (1368 pages: 7 NRC reports, 4 CC BY PHYSOR papers, the
maintainer's arXiv paper):

| layer | 0.4.1 | 0.4.2 |
|---|---|---|
| object kinds (42,771 objects) | 0 mismatches | 0 mismatches |
| non-image streams byte-identical | all | all (the 503 differing streams are image codecs: CCITT 237, JPX 211, DCT 50, JBIG2 5 -- `open_stream` stops before the codec, `pdf_load_stream` runs it; the decoded images are compared by the raster layer) |
| text pages passing | 1245 / 1368 | **1368 / 1368** |
| chars missing / extra (of 2,023,821) | 974 / 412 | **0 / 0** |
| reading-order mismatches | 93 pages | **0** |
| raster pages within 1 % gross | 1099 / 1368 | **1368 / 1368** (worst page 0.34 %, WASH-1400 pixel-identical) |

**Synthetic feature corpus** (`scripts/mupdf-feature-corpus.py`, 34 files,
one feature each): **33 / 34 pass** (0.4.1: 2 of the first 24). All but four
are pixel-identical or 0.00 % gross. The four with a residual:

- `text-render-modes` 1.84 % (the one failure)
- `tiling-pattern-2` 0.88 %
- `text-transparency` 0.84 %
- `cmyk-fill` 0.00 % gross, but not pixel-identical

The first three share one cause: glyph shape. The non-embedded
`Helvetica-Bold` is drawn from our bundled base-14 face, while MuPDF uses
Nimbus Sans. Plain text in that font measures 0.875 %.

## Appendix: per-file function counts

Columns:

* **Functions defined**: the column-0 definition count from the method above.
* **Named in kopitiam-pdf**: how many of those names appear in `crates/kopitiam-pdf/src/**/*.rs` under the attribution rule.
* **Rust cites the file?**: whether any `.rs` names the C file's basename.
* **C lines**: `wc -l`.

| MuPDF file | C lines | Functions defined | Named in kopitiam-pdf | % | Rust cites the file? | Status |
|---|---:|---:|---:|---:|:---:|---|
| `fitz/archive.c` | 576 | 32 | 0 | 0 | no | out-of-scope |
| `fitz/barcode.c` | 225 | 7 | 0 | 0 | no | out-of-scope |
| `fitz/bbox-device.c` | 228 | 21 | 0 | 0 | no | out-of-scope |
| `fitz/bidi-std.c` | 1173 | 20 | 0 | 0 | no | missing |
| `fitz/bidi.c` | 865 | 9 | 0 | 0 | no | missing |
| `fitz/bitmap.c` | 582 | 19 | 0 | 0 | no | out-of-scope |
| `fitz/brotli.c` | 139 | 6 | 0 | 0 | no | out-of-scope |
| `fitz/buffer.c` | 681 | 41 | 32 | 78 | yes | translated |
| `fitz/color-fast.c` | 1736 | 34 | 0 | 0 | no | partial |
| `fitz/color-icc-create.c` | 641 | 27 | 0 | 0 | no | missing |
| `fitz/color-lcms.c` | 535 | 19 | 0 | 0 | no | missing |
| `fitz/colorspace.c` | 2016 | 95 | 2 | 2 | yes | partial |
| `fitz/compress.c` | 106 | 4 | 0 | 0 | no | out-of-scope |
| `fitz/compressed-buffer.c` | 179 | 7 | 1 | 14 | no | ~~partial~~ translated (0.4.2) |
| `fitz/context.c` | 399 | 24 | 0 | 0 | no | substituted-by-design |
| `fitz/crypt-aes.c` | 578 | 6 | 0 | 0 | no | substituted-by-design |
| `fitz/crypt-arc4.c` | 104 | 4 | 0 | 0 | no | substituted-by-design |
| `fitz/crypt-md5.c` | 283 | 5 | 0 | 0 | no | substituted-by-design |
| `fitz/crypt-sha2.c` | 409 | 14 | 0 | 0 | no | ~~missing~~ substituted-by-design (0.4.2) |
| `fitz/cull-device.c` | 581 | 28 | 0 | 0 | no | out-of-scope |
| `fitz/deskew.c` | 1194 | 15 | 0 | 0 | no | out-of-scope |
| `fitz/device.c` | 1191 | 80 | 5 | 6 | yes | partial |
| `fitz/directory.c` | 252 | 9 | 0 | 0 | no | out-of-scope |
| `fitz/document-all.c` | 80 | 1 | 0 | 0 | no | out-of-scope |
| `fitz/document.c` | 1324 | 78 | 1 | 1 | no | out-of-scope |
| `fitz/draw-affine.c` | 4124 | 242 | 3 | 1 | yes | ~~partial~~ translated (0.4.2) |
| `fitz/draw-blend.c` | 1369 | 26 | 0 | 0 | yes | ~~missing~~ translated (0.4.2) |
| `fitz/draw-device.c` | 3501 | 69 | 5 | 7 | yes | partial |
| `fitz/draw-edge.c` | 918 | 24 | 6 | 25 | yes | partial |
| `fitz/draw-edgebuffer.c` | 1922 | 40 | 0 | 0 | no | missing |
| `fitz/draw-glyph.c` | 494 | 14 | 0 | 0 | yes | missing |
| `fitz/draw-mesh.c` | 517 | 8 | 0 | 0 | yes | ~~missing~~ translated (0.4.2) |
| `fitz/draw-paint.c` | 3212 | 136 | 0 | 0 | yes | partial |
| `fitz/draw-path.c` | 1628 | 46 | 14 | 30 | yes | partial |
| `fitz/draw-rasterize.c` | 311 | 21 | 0 | 0 | no | partial |
| `fitz/draw-scale-simple.c` | 1897 | 30 | 0 | 0 | no | ~~missing~~ translated (0.4.2) |
| `fitz/draw-unpack.c` | 514 | 14 | 1 | 7 | no | translated |
| `fitz/encode-basic.c` | 492 | 27 | 0 | 0 | no | out-of-scope |
| `fitz/encode-fax.c` | 318 | 8 | 0 | 0 | no | out-of-scope |
| `fitz/encode-jpx.c` | 379 | 13 | 0 | 0 | no | out-of-scope |
| `fitz/encodings.c` | 205 | 11 | 3 | 27 | yes | translated |
| `fitz/error.c` | 563 | 30 | 8 | 27 | yes | translated |
| `fitz/filter-basic.c` | 891 | 31 | 11 | 35 | yes | translated |
| `fitz/filter-brotli.c` | 136 | 5 | 0 | 0 | no | missing |
| `fitz/filter-dct.c` | 410 | 15 | 0 | 0 | no | substituted-by-design |
| `fitz/filter-fax.c` | 854 | 12 | 9 | 75 | yes | translated |
| `fitz/filter-flate.c` | 145 | 5 | 2 | 40 | yes | substituted-by-design |
| `fitz/filter-jbig2.c` | 257 | 12 | 0 | 0 | no | ~~missing~~ substituted-by-design (0.4.2) |
| `fitz/filter-leech.c` | 76 | 3 | 1 | 33 | no | out-of-scope |
| `fitz/filter-lzw.c` | 269 | 3 | 2 | 67 | yes | translated |
| `fitz/filter-predict.c` | 305 | 8 | 7 | 88 | yes | translated |
| `fitz/filter-sgi.c` | 683 | 13 | 0 | 0 | no | out-of-scope |
| `fitz/filter-thunder.c` | 163 | 3 | 0 | 0 | no | out-of-scope |
| `fitz/font.c` | 2468 | 89 | 10 | 11 | yes | substituted-by-design |
| `fitz/ftoa.c` | 301 | 10 | 1 | 10 | no | substituted-by-design |
| `fitz/geometry.c` | 1086 | 73 | 50 | 68 | yes | translated |
| `fitz/getopt.c` | 251 | 6 | 0 | 0 | no | out-of-scope |
| `fitz/glyph.c` | 471 | 11 | 0 | 0 | yes | missing |
| `fitz/glyphbox.c` | 39 | 1 | 0 | 0 | no | out-of-scope |
| `fitz/gz-doc.c` | 105 | 2 | 0 | 0 | no | out-of-scope |
| `fitz/halftone.c` | 655 | 12 | 0 | 0 | no | out-of-scope |
| `fitz/harfbuzz.c` | 189 | 8 | 0 | 0 | no | out-of-scope |
| `fitz/hash.c` | 329 | 11 | 10 | 91 | yes | translated |
| `fitz/heap.c` | 31 | 0 | 0 | – | no | substituted-by-design |
| `fitz/hyphen.c` | 407 | 12 | 0 | 0 | no | out-of-scope |
| `fitz/image.c` | 1839 | 63 | 1 | 2 | yes | partial |
| `fitz/jmemcust.c` | 158 | 10 | 0 | 0 | no | substituted-by-design |
| `fitz/json.c` | 779 | 46 | 0 | 0 | no | out-of-scope |
| `fitz/leptonica-wrap.c` | 107 | 6 | 0 | 0 | no | out-of-scope |
| `fitz/link.c` | 106 | 6 | 0 | 0 | yes | translated |
| `fitz/list-device.c` | 2131 | 43 | 0 | 0 | no | missing |
| `fitz/list.c` | 123 | 4 | 0 | 0 | yes | substituted-by-design |
| `fitz/load-bmp.c` | 1512 | 19 | 0 | 0 | no | out-of-scope |
| `fitz/load-gif.c` | 649 | 18 | 0 | 0 | no | out-of-scope |
| `fitz/load-jbig2.c` | 219 | 10 | 0 | 0 | no | out-of-scope |
| `fitz/load-jpeg.c` | 576 | 18 | 1 | 6 | yes | substituted-by-design |
| `fitz/load-jpx.c` | 699 | 27 | 0 | 0 | no | ~~missing~~ substituted-by-design (0.4.2) |
| `fitz/load-jxr.c` | 484 | 11 | 0 | 0 | no | out-of-scope |
| `fitz/load-png.c` | 699 | 17 | 0 | 0 | no | out-of-scope |
| `fitz/load-pnm.c` | 1062 | 27 | 0 | 0 | no | out-of-scope |
| `fitz/load-psd.c` | 513 | 9 | 0 | 0 | no | out-of-scope |
| `fitz/load-tiff.c` | 1820 | 32 | 0 | 0 | no | out-of-scope |
| `fitz/log.c` | 89 | 3 | 0 | 0 | no | out-of-scope |
| `fitz/memento.c` | 4812 | 141 | 0 | 0 | no | out-of-scope |
| `fitz/memory.c` | 509 | 29 | 1 | 3 | no | substituted-by-design |
| `fitz/noto.c` | 557 | 18 | 0 | 0 | no | partial |
| `fitz/ocr-device.c` | 1025 | 44 | 0 | 0 | no | out-of-scope |
| `fitz/options.c` | 723 | 32 | 0 | 0 | no | out-of-scope |
| `fitz/outline.c` | 322 | 22 | 0 | 0 | yes | translated |
| `fitz/output-cbz.c` | 236 | 11 | 0 | 0 | no | out-of-scope |
| `fitz/output-csv.c` | 345 | 13 | 0 | 0 | no | out-of-scope |
| `fitz/output-docx.c` | 895 | 32 | 0 | 0 | no | out-of-scope |
| `fitz/output-jpeg.c` | 319 | 13 | 0 | 0 | no | out-of-scope |
| `fitz/output-pcl.c` | 1584 | 34 | 0 | 0 | no | out-of-scope |
| `fitz/output-pclm.c` | 445 | 19 | 0 | 0 | no | out-of-scope |
| `fitz/output-pdfocr.c` | 1184 | 28 | 0 | 0 | no | out-of-scope |
| `fitz/output-png.c` | 429 | 12 | 0 | 0 | no | out-of-scope |
| `fitz/output-pnm.c` | 486 | 20 | 0 | 0 | no | out-of-scope |
| `fitz/output-ps.c` | 413 | 15 | 0 | 0 | no | out-of-scope |
| `fitz/output-psd.c` | 469 | 11 | 0 | 0 | no | out-of-scope |
| `fitz/output-pwg.c` | 666 | 24 | 0 | 0 | no | out-of-scope |
| `fitz/output-svg.c` | 130 | 5 | 0 | 0 | no | out-of-scope |
| `fitz/output.c` | 885 | 66 | 1 | 2 | yes | out-of-scope |
| `fitz/path.c` | 2098 | 56 | 9 | 16 | yes | partial |
| `fitz/pixmap.c` | 2198 | 59 | 7 | 12 | yes | partial |
| `fitz/pool.c` | 218 | 11 | 10 | 91 | yes | translated |
| `fitz/printf.c` | 840 | 23 | 0 | 0 | yes | substituted-by-design |
| `fitz/random.c` | 111 | 11 | 0 | 0 | no | out-of-scope |
| `fitz/regexp.c` | 34 | 0 | 0 | – | no | out-of-scope |
| `fitz/separation.c` | 1251 | 20 | 0 | 0 | no | missing |
| `fitz/shade.c` | 1139 | 33 | 0 | 0 | no | ~~missing~~ translated (0.4.2) |
| `fitz/skew.c` | 266 | 7 | 0 | 0 | no | out-of-scope |
| `fitz/stext-boxer.c` | 1037 | 29 | 19 | 66 | yes | translated |
| `fitz/stext-classify.c` | 462 | 5 | 2 | 40 | yes | partial |
| `fitz/stext-device.c` | 3117 | 88 | 13 | 15 | yes | partial |
| `fitz/stext-iterator.c` | 253 | 15 | 2 | 13 | yes | translated |
| `fitz/stext-output.c` | 1518 | 46 | 1 | 2 | no | out-of-scope |
| `fitz/stext-para.c` | 1635 | 45 | 10 | 22 | yes | partial |
| `fitz/stext-raft.c` | 453 | 13 | 0 | 0 | no | missing |
| `fitz/stext-search.c` | 2130 | 73 | 6 | 8 | yes | partial |
| `fitz/stext-table.c` | 4623 | 82 | 0 | 0 | yes | missing |
| `fitz/store.c` | 1126 | 28 | 0 | 0 | no | substituted-by-design |
| `fitz/stream-open.c` | 387 | 20 | 5 | 25 | yes | translated |
| `fitz/stream-read.c` | 612 | 30 | 8 | 27 | yes | translated |
| `fitz/string.c` | 1264 | 47 | 21 | 45 | yes | translated |
| `fitz/strtof.c` | 481 | 9 | 1 | 11 | no | substituted-by-design |
| `fitz/subset-cff.c` | 2482 | 45 | 3 | 7 | yes | partial |
| `fitz/subset-ttf.c` | 2039 | 43 | 0 | 0 | no | out-of-scope |
| `fitz/svg-device.c` | 1532 | 55 | 0 | 0 | no | out-of-scope |
| `fitz/test-device.c` | 416 | 14 | 0 | 0 | no | out-of-scope |
| `fitz/text-decoder.c` | 236 | 17 | 0 | 0 | no | out-of-scope |
| `fitz/text.c` | 308 | 13 | 1 | 8 | yes | partial |
| `fitz/time.c` | 277 | 10 | 0 | 0 | no | out-of-scope |
| `fitz/trace-device.c` | 709 | 42 | 0 | 0 | no | out-of-scope |
| `fitz/track-usage.c` | 56 | 2 | 0 | 0 | no | out-of-scope |
| `fitz/transition.c` | 225 | 6 | 0 | 0 | no | out-of-scope |
| `fitz/tree.c` | 137 | 6 | 0 | 0 | no | substituted-by-design |
| `fitz/ucdn.c` | 361 | 25 | 1 | 4 | no | substituted-by-design |
| `fitz/uncfb.c` | 832 | 27 | 0 | 0 | no | out-of-scope |
| `fitz/unlibarchive.c` | 716 | 23 | 0 | 0 | no | out-of-scope |
| `fitz/untar.c` | 354 | 16 | 0 | 0 | no | out-of-scope |
| `fitz/unzip.c` | 720 | 17 | 0 | 0 | no | out-of-scope |
| `fitz/util.c` | 1197 | 54 | 1 | 2 | no | partial |
| `fitz/warp.c` | 2313 | 45 | 0 | 0 | no | out-of-scope |
| `fitz/writer.c` | 374 | 29 | 0 | 0 | no | out-of-scope |
| `fitz/xml-write.c` | 191 | 6 | 0 | 0 | no | out-of-scope |
| `fitz/xml.c` | 1430 | 57 | 0 | 0 | no | out-of-scope |
| `fitz/xmltext-device.c` | 404 | 19 | 0 | 0 | no | out-of-scope |
| `fitz/zip.c` | 163 | 5 | 0 | 0 | no | out-of-scope |
| `pdf/pdf-af.c` | 52 | 4 | 0 | 0 | no | missing |
| `pdf/pdf-annot.c` | 4654 | 195 | 17 | 9 | yes | partial |
| `pdf/pdf-appearance.c` | 3931 | 99 | 15 | 15 | yes | partial |
| `pdf/pdf-clean-file.c` | 618 | 12 | 0 | 0 | no | out-of-scope |
| `pdf/pdf-clean.c` | 1398 | 31 | 0 | 0 | no | out-of-scope |
| `pdf/pdf-cmap-load.c` | 338 | 5 | 2 | 40 | yes | partial |
| `pdf/pdf-cmap-parse.c` | 444 | 13 | 11 | 85 | yes | translated |
| `pdf/pdf-cmap.c` | 965 | 26 | 12 | 46 | yes | translated |
| `pdf/pdf-colorspace.c` | 748 | 17 | 1 | 6 | yes | partial |
| `pdf/pdf-crypt.c` | 1517 | 46 | 0 | 0 | yes | partial |
| `pdf/pdf-device.c` | 1436 | 48 | 0 | 0 | no | out-of-scope |
| `pdf/pdf-event.c` | 164 | 11 | 0 | 0 | no | out-of-scope |
| `pdf/pdf-font-add.c` | 826 | 20 | 0 | 0 | no | out-of-scope |
| `pdf/pdf-font.c` | 1699 | 41 | 9 | 22 | yes | partial |
| `pdf/pdf-form.c` | 2595 | 123 | 16 | 13 | yes | partial |
| `pdf/pdf-function.c` | 1568 | 37 | 0 | 0 | no | ~~missing~~ translated (0.4.2) |
| `pdf/pdf-graft.c` | 293 | 7 | 0 | 0 | no | out-of-scope |
| `pdf/pdf-image-rewriter.c` | 941 | 14 | 0 | 0 | no | out-of-scope |
| `pdf/pdf-image.c` | 853 | 12 | 2 | 17 | yes | partial |
| `pdf/pdf-interpret.c` | 2094 | 132 | 13 | 10 | yes | partial |
| `pdf/pdf-js.c` | 1346 | 78 | 0 | 0 | yes | out-of-scope |
| `pdf/pdf-label.c` | 265 | 8 | 0 | 0 | no | out-of-scope |
| `pdf/pdf-layer.c` | 945 | 29 | 1 | 3 | no | ~~missing~~ partial (0.4.2) |
| `pdf/pdf-layout.c` | 225 | 4 | 0 | 0 | no | missing |
| `pdf/pdf-lex.c` | 736 | 19 | 13 | 68 | yes | translated |
| `pdf/pdf-link.c` | 1508 | 46 | 4 | 9 | yes | partial |
| `pdf/pdf-metrics.c` | 166 | 11 | 3 | 27 | yes | translated |
| `pdf/pdf-nametree.c` | 384 | 10 | 1 | 10 | no | partial |
| `pdf/pdf-object.c` | 4363 | 237 | 61 | 26 | yes | translated |
| `pdf/pdf-op-buffer.c` | 1549 | 110 | 0 | 0 | no | out-of-scope |
| `pdf/pdf-op-color.c` | 1315 | 49 | 0 | 0 | no | out-of-scope |
| `pdf/pdf-op-filter.c` | 3129 | 130 | 0 | 0 | no | out-of-scope |
| `pdf/pdf-op-run.c` | 3591 | 153 | 38 | 25 | yes | partial |
| `pdf/pdf-op-vectorize.c` | 638 | 38 | 0 | 0 | no | out-of-scope |
| `pdf/pdf-outline.c` | 597 | 13 | 2 | 15 | yes | translated |
| `pdf/pdf-page.c` | 2030 | 82 | 8 | 10 | yes | partial |
| `pdf/pdf-parse.c` | 982 | 25 | 12 | 48 | yes | translated |
| `pdf/pdf-pattern.c` | 114 | 5 | 0 | 0 | no | ~~missing~~ translated (0.4.2) |
| `pdf/pdf-recolor.c` | 177 | 7 | 0 | 0 | no | out-of-scope |
| `pdf/pdf-repair.c` | 1016 | 15 | 0 | 0 | yes | ~~missing~~ translated (0.4.2) |
| `pdf/pdf-resources.c` | 199 | 12 | 0 | 0 | yes | out-of-scope |
| `pdf/pdf-run.c` | 691 | 19 | 5 | 26 | yes | partial |
| `pdf/pdf-shade-recolor.c` | 1030 | 15 | 0 | 0 | no | out-of-scope |
| `pdf/pdf-shade.c` | 535 | 15 | 0 | 0 | no | ~~missing~~ translated (0.4.2) |
| `pdf/pdf-signature.c` | 668 | 24 | 0 | 0 | no | missing |
| `pdf/pdf-store.c` | 159 | 14 | 0 | 0 | no | substituted-by-design |
| `pdf/pdf-stream.c` | 895 | 32 | 13 | 41 | yes | partial |
| `pdf/pdf-struct.c` | 248 | 6 | 0 | 0 | no | missing |
| `pdf/pdf-subset.c` | 793 | 24 | 0 | 0 | no | out-of-scope |
| `pdf/pdf-type3.c` | 247 | 4 | 0 | 0 | no | ~~missing~~ translated (0.4.2) |
| `pdf/pdf-unicode.c` | 157 | 3 | 3 | 100 | yes | translated |
| `pdf/pdf-util.c` | 232 | 7 | 0 | 0 | no | out-of-scope |
| `pdf/pdf-write.c` | 3320 | 77 | 3 | 4 | yes | partial |
| `pdf/pdf-xobject.c` | 133 | 9 | 1 | 11 | yes | partial |
| `pdf/pdf-xref.c` | 5550 | 160 | 19 | 12 | yes | partial |
| `pdf/pdf-zugferd.c` | 277 | 4 | 0 | 0 | no | out-of-scope |
| **total (205 files)** | **191030** | **6380** | **593** | **9.3** | | |

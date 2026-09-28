# Scanned paper open blank: `/Contents` point to an array, and the OCR layer got painted

**Date:** 2026-09-28
**About:** `crates/kopitiam-pdf/src/mupdf/page_run.rs` (`gather_contents`), `draw_device.rs` / `text_device.rs` / `op_run.rs` (text render mode), `standard_font.rs` (stage-3 substitution); kopitiam-pdf 0.4.1
**Related:** [The guard you drop when porting is the one you never had a fixture for](2026-08-25-dropped-guards-in-a-port.md)

## What happen

Maintainer open a 1973 journal paper in kovan: every page blank, "cannot see
characters". The file is an **Acrobat Capture 3.0 "searchable image"** scan:
the visible page is ~26 CCITT-G4 strip images (2250x115 px each, one `Do` per
strip), and on top sit an **invisible (`3 Tr`) OCR text layer** in
**non-embedded** TrueType fonts (`TimesNewRoman`, `Arial,Bold`,
`BookAntiqua,BoldItalic`, `CenturyGothic`, `LetterGothicMT`, ...). poppler show
it fine. `kpdf-doctor --render` said `page is completely blank (0 dark
pixels)`, render time 0.5 ms — i.e. the content stream never ran at all.

Three stacked defects, found one after the other lah:

1. **`/Contents 1169 0 R` where object 1169 is an ARRAY** of 27 tiny LZW
   streams. §7.7.3.3 allow this. The port matched on the *unresolved* object:
   a direct array got concatenated, but a `Ref` went straight to
   `open_stream`, which failed "not a stream"; the error was swallowed and the
   page ran with **zero bytes**. MuPDF never had this bug because
   `pdf_open_contents_stream` tests `pdf_is_array`, and `pdf_is_array`
   *resolves first*. Same lesson as the #70 entry: the port kept the formula,
   lost the control flow, and no fixture ever had an indirect array.
   Blank page, no image, no text, no error — silent wrongness again.
2. **Invisible text got painted.** Once the page ran, the OCR words drew
   over the scan: substituted outlines slightly off the scanned letters, and
   solid advance boxes for faces with no substitute. MuPDF's
   `pdf_flush_text_imp` fills nothing for mode 3 (`doinvisible`) or 7
   (`doclip`); our draw device painted every glyph whatever `Tr` said. Worse,
   the boxes counted as fallback glyphs, so `rasterize_page` threw the whole
   page to hayro for no reason. Fix: `TextDevice::set_text_render_mode`
   (default no-op, same shape as `set_fill_color`); the draw device skips 3
   and 7, the extraction devices still see every glyph — that invisible layer
   is the only text a scan has.
3. **Unknown names with a text descriptor got no substitute.**
   `BookAntiqua,BoldItalic` has no family keyword, so `select_standard_font`
   returned `None` → boxes (or nothing, if `/Widths` absent). MuPDF's last
   resort (`pdf_lookup_substitute_font`) picks Courier/Times/Helvetica from
   the descriptor's FixedPitch/Serif flags and never gives up. We now do the
   same, but only when `/Flags` says **Nonsymbolic and not Symbolic** — the
   deliberate "Wingdings-as-Helvetica is confident nonsense" refusal stays for
   symbolic or flag-less fonts.

## What to remember

* **Resolve before you branch on type.** Any `match obj { Object::Array(..) }`
  on a value that PDF allows to be indirect is a latent version of defect 1.
  MuPDF's `pdf_is_*` predicates all resolve; a port that pattern-matches the
  raw object does not, and the difference only shows on files whose producer
  likes indirect objects (Capture, some old Distiller output).
* **"Render" and "extract" want different things from the same glyph.** Text
  render mode is a painting concern only. Put it on the painting device, never
  filter glyphs in the interpreter, or extraction of scanned papers dies.
* A doctor line of "0 dark pixels" + sub-millisecond render time means the
  content stream did not run — look at `/Contents` before fonts or images.

## Follow-ups noticed, not fixed here

* 1-bit scan strips look bolder than poppler at 72 dpi (dark-pixel count
  ~13% higher on the page checked) — downscale filtering of `ImageMask`/1-bpc
  images, not a correctness bug; readable.
* `kpdf-doctor`'s "suspiciously solid black" check fires on dense 1-bit text
  scans (5–7% pure black). False positive for this class of document.

Tests: `crates/kopitiam-pdf/tests/scanned_ocr_layer.rs` (synthetic
Capture-shaped fixtures; the paper itself is restricted literature and is not
a fixture) + `standard_font.rs` unit tests for the flag stage.

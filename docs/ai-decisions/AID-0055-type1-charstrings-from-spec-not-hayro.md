# AID-0055: Type1 charstring decoding is a from-spec implementation, not a dependency on `hayro-font`

* **Status:** Pending review
* **Date:** 2026-08-24
* **Decided by:** AI (Claude), maintainer absent (but see "The prompt" below)
* **Scope:** `crates/kopitiam-pdf/src/mupdf/glyph_type1.rs` (new) — the Type1
  (`/FontFile`) glyph-outline decoder that closes GitHub issue
  `theodoreOnzGit/kopitiam#31`. Extends AID-0052 (the port's "substitute a
  pure-Rust crate, or re-implement from spec, at every point MuPDF delegates
  to a C library" rule) to a third font format alongside TrueType and CFF.

## The prompt

Issue #31 reports that Type1 (`/FontFile`) and predefined-encoding simple CFF
text renders as solid filled boxes — the documented ceiling in
`glyph.rs`/`glyph_cff.rs` at the time. The issue explicitly declined to
prescribe a fix ("happy to discuss scope/priority"), flagging it as "a real
chunk of work (a second charstring interpreter)". Mid-implementation the
maintainer sent one line: "you may depend on haryo if you want" — a real-time
steer, not a full brief, so the actual dependency-or-not call, and everything
about how to structure the interpreter, was still mine to make.

## The decision

Two decisions, both executed:

### 1. Implement the Type1 interpreter from the Adobe Type 1 Font Format spec, not by depending on `hayro-font`

`hayro-font` (crates.io, part of the `hayro` PDF-rasterizer project by
LaurenzV, `Apache-2.0 OR MIT`) does exist and does parse "CFF and Type1
fonts" — confirmed via its crates.io listing. But its own docs.rs page states
plainly: *"an internal crate and not meant to be used directly. Therefore,
it's not well-documented."* That is a maintained project declining to offer a
public API contract for this crate — no SemVer promise governs its shape, and
a minor `hayro` bump could rename or restructure it under us with no warning.

Taking it as a dependency would also cut against the pattern AID-0052 already
set for this exact problem shape (TrueType via the OpenType spec, CFF via
Adobe TN#5177) and the reason given there: FreeType has no *clean* pure-Rust
drop-in the port wants to own as a dependency, so the port re-implements
outline decoding from the public specification instead. `hayro-font`
explicitly is not offered as that drop-in.

So `glyph_type1.rs` is a clean-room implementation against the **Adobe Type 1
Font Format specification** (Adobe Systems Inc., 1990) — the same document
FreeType's `psaux`/`type1` driver and `t1lib` implement, and the one MuPDF
itself never re-implements (MuPDF's `/FontFile` path always goes through
FreeType, so there is no MuPDF C to translate here — same situation AID-0052
already documents for the TrueType/CFF decoders). Per the project's clean-room
attribution rules, the module doc cites the spec by section number
(6.2 charstring number encoding, 7.2/7.3 the two decryption ciphers, 8.3 flex
+ hint replacement, 8.7 `seac`) rather than any implementation's source.

### 2. Cross-check the CFF Standard Strings table against a canonical source, not memory

The predefined-Standard-CFF-encoding fix (the issue's other named case)
needed the 391-entry CFF Standard Strings table (Adobe TN#5176 Appendix A) —
a long, easy-to-mistranscribe, purely mechanical list. Rather than transcribe
it from training-data recall, it was fetched and cross-checked against
fontTools' `cffLib.cffStandardStrings` (BSD-3-Clause) via `WebFetch`, counted
programmatically (391 entries, `.notdef`…`Semibold`) before being written into
`glyph_cff.rs`. This is the "vendor references come first, don't invent from
scratch" hard rule applied to a data table rather than an algorithm — the
list itself is a fixed part of the CFF spec (not fontTools' creative
expression), so citing fontTools is for verification/provenance, not because
the table is fontTools' to attribute.

## Alternatives considered

* **Depend on `hayro-font` directly**, per the maintainer's permission. Ruled
  out for the "internal, not for direct use" reason above — a decade-horizon
  project taking a hard dependency on an admittedly-unstable internal API is
  a worse trade than ~700 lines of spec-based, tested, self-owned code,
  especially given the existing TrueType/CFF decoders already prove the
  pattern works here.
* **Vendor `hayro`'s Type1 source as a `vendor/` reference** (per the
  "vendored references come first" rule) rather than either depending on it
  or working from memory. Attempted: `hayro-font` did not resolve at the
  `crates/hayro-font/src/type1/mod.rs`-shaped paths tried against the
  `LaurenzV/hayro` GitHub tree (404s; the crate may have been merged into
  `hayro-interpret` or restructured since its last publish), and chasing the
  actual path further had a poor cost/benefit against the time already spent
  confirming the spec-based constants (the eexec/charstring cipher constants
  57665/4330/52845/22719 and the `OtherSubrs` 0–3 convention are fixed,
  decades-old, and independently well-established across FreeType, t1lib,
  and Ottheus PDF codebases). Not vendored; spec citation only.
* **Skip predefined-Expert CFF encoding too** (only Standard was implemented).
  Kept as a documented ceiling (falls back to the advance box) — Expert-set
  fonts are rare in practice, and the issue's own report was specifically
  about Type1 and (implicitly, per its title) the far more common Standard
  case.

## What would make this wrong

* If `hayro-font` (or another pure-Rust Type1 decoder) later ships a stable,
  documented public API with a SemVer contract, revisit — the "internal
  crate" objection would no longer hold, and a maintained upstream doing the
  same job is generally preferable to an equivalent amount of self-owned code
  once it's safe to lean on.
* If a real-world PDF's Type1 glyphs render subtly wrong (bad flex curves, a
  `seac` accent placed off, hint-replacement side effects leaking into the
  outline), that is evidence the from-spec reasoning in
  `glyph_type1.rs`'s `callothersubr`/`seac` handling has a bug worth
  cross-checking against a second implementation (FreeType's `t1decode.c` or
  `hayro-font`, wherever it currently lives) rather than re-deriving from the
  spec text again.
* If the CFF Standard Strings table is ever found to mismatch the Adobe
  TN#5176 spec at some index, that is a transcription bug in this table
  specifically (it was cross-checked once, not re-verified per release) —
  worth a fresh WebFetch-and-diff against fontTools or the spec PDF directly.

## Relationship to AID-0052

AID-0052 established the rule this decision applies: at every point MuPDF
delegates to a C library it doesn't itself implement, substitute a mature
pure-Rust crate where one *cleanly* exists, or re-implement from the public
format spec where it doesn't. Type1 is the third and, with predefined-CFF
now covered too, the port's embedded-font glyph-outline coverage is now
TrueType `glyf` + CFF/Type2 (custom and predefined-Standard encoding) + Type1
— predefined-Expert CFF is the one remaining documented gap.

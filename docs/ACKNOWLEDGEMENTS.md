# Acknowledgements

**KOPITIAM is licensed AGPL-3.0-only, in its entirety, without exception.**
Everything listed on this page that KOPITIAM forks, translates, adapts, or
merely learns from is credited to its upstream authors below. Attribution is a
hard rule in this project (see `CLAUDE.md`, "Working Practices"), and it
distinguishes carefully between three different relationships:

| Relationship | Obligation |
| --- | --- |
| **Clean-room study** — read the papers/docs/APIs, understand the algorithm, write original Rust | Credit the project here. No code is copied. |
| **Translation / close adaptation** of a specific algorithm | Credit here **and** name the source in a doc comment *at the point of use*. |
| **Fork / direct code reuse** | Retain upstream copyright notices and license text; state plainly in the crate rustdoc that it is a fork, of what, under what license. |

Knowing which of the three you are doing is not a formality. Conflating them is
how a project acquires a licensing problem it cannot unwind later.

---

## Reference projects (clean-room study)

KOPITIAM's local-first inference runtime (the long-term implementation
behind the `kopitiam-ai` `ModelAdapter` boundary — see `crates/kopitiam-ai`
and `CLAUDE.md`'s Semantic Runtime section) is an independent, from-scratch
Rust implementation. It is **not** a fork or port of any of the projects below.
They are studied for architecture, algorithms, and file formats, and cloned
locally as reference material — never built, linked, or shipped as part of
KOPITIAM.

| Project | License | Studied for |
| --- | --- | --- |
| [Candle](https://github.com/huggingface/candle) | Apache-2.0 OR MIT | Rust tensor design, transformer implementation, model loading |
| [Burn](https://github.com/tracel-ai/burn) | MIT OR Apache-2.0 | Backend abstraction, modular training/inference architecture |
| [ggml](https://github.com/ggml-org/ggml) | MIT | Tensor kernels, quantization, KV cache, CPU execution, GGUF format |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | Qwen support, GGUF loading, sampling, CPU optimization, scheduler, memory layout |
| [SafeTensors](https://github.com/huggingface/safetensors) | Apache-2.0 | Weight serialization, memory mapping |
| [Tokenizers](https://github.com/huggingface/tokenizers) | Apache-2.0 | Rust-native tokenizer design |
| [TensorFlow](https://github.com/tensorflow/tensorflow) | Apache-2.0 | Reference only — graph execution, operator design |
| [XNNPACK](https://github.com/google/XNNPACK) | BSD-3-Clause | CPU operators, SIMD kernels, matmul optimization |
| [oneDNN](https://github.com/oneapi-src/oneDNN) | Apache-2.0 | Linear algebra and kernel optimization, operator fusion |
| [ONNX](https://github.com/onnx/onnx) | Apache-2.0 | Model interchange format, for possible future ONNX support |
| [Neovim](https://github.com/neovim/neovim) | Apache-2.0, plus Vim-licensed portions | Editor architecture, the `vim.*` API surface, and modal-editing semantics, for `kopitiam-neovim` (`kvim`) |
| [transformers](https://github.com/huggingface/transformers) (HuggingFace) | Apache-2.0 | The reference **model implementations** (LLaMA-shaped architectures, attention / RoPE / RMSNorm layouts, tokenizer configs) studied for `kopitiam-runtime`'s from-scratch inference. A read-only reference clone is vendored at **`crates/kopitiam-runtime/vendor/transformers`** — gitignored, never built, linked, or shipped |
| [ollama](https://github.com/ollama/ollama) | MIT | The reference **local model-serving runtime**. Studied for the sampling defaults a local chat must run with: `api/types.go`'s `DefaultOptions()` (temperature 0.8, top_k 40, top_p 0.9, repeat_penalty 1.1, repeat_last_n 64) is transcribed into `kopitiam-ai`'s `local::generation::default_sampling`, with the one deliberate divergence — a pinned PRNG seed, where ollama defaults `Seed: -1` to entropy — documented at the point of use. Parameter values only, no code (ollama is Go). A read-only reference clone is vendored at **`crates/kopitiam-runtime/vendor/ollama`** — gitignored, never built, linked, or shipped |
| [lazygit](https://github.com/jesseduffield/lazygit) (Jesse Duffield) | MIT | **UI reference** for the CLI's git-panel workflow — how a mature terminal git UI lays out status / staging / branch / log panes and their keybindings. Studied for the panel UX only, not for code (lazygit is Go). A read-only reference clone is vendored at **`apps/cli/vendor/lazygit`** — gitignored, never built or shipped |
| [Helix](https://github.com/helix-editor/helix) | MPL-2.0 | Modal-editor **infrastructure and feature-completeness reference** for `kvim` — how a mature Rust editor wires LSP lifecycle, incremental syntax, a command palette, and buffer/window management. **kvim is vim-modeled**, so Helix's selection-first keymap is studied for *what* mature editors do, never for *how* kvim binds keys. Clean-room study only: no Helix code is copied, and MPL-2.0 governs any file that ever were — none is. |
| [tmux](https://github.com/tmux/tmux) | ISC | The **terminal-multiplexer behavioural reference** for `kmux`. Read as data only — **no code copied** (tmux is C; `kmux` descends from rmux, see the fork table below). Consulted to settle exactly which lifecycle notifications a forced respawn may emit: `server-fn.c`'s `pane-exited` fires only from `server_destroy_pane()` (the child-death path), while `spawn.c`'s `SPAWN_RESPAWN\|SPAWN_KILL` path notifies nothing, and `input.c`'s `pane-title-changed` fires purely off the child writing OSC 0/2 with nothing in respawn suppressing it. That distinction is asserted in `crates/kmux/src/server/handler_pane_command_tests.rs` and attributed at the point of use. A read-only reference clone is vendored at **`crates/kmux/vendor/tmux`** (pinned `5534f1ac32`) — gitignored, never built, linked, or shipped |
| [Language Server Protocol specification](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/) | CC-BY-4.0 (spec text) | The **snippet syntax grammar** implemented clean-room by `kopitiam-snippet` (`$1`, `${1:…}`, `${1\|a,b\|}`, `$VAR`, escapes, mirrors). Only the published *grammar* is followed; **no code is copied** from LuaSnip, vsnip, or VS Code. This is a specification reference, not a source-code reference. |
| [Lua](https://www.lua.org/) 5.1 (PUC-Rio) — the [Lua 5.1 Reference Manual](https://www.lua.org/manual/5.1/) | MIT | The **Lua 5.1 language** implemented clean-room by `kopitiam-lua` (`kopitiam-lua` is a pure-Rust Lua 5.1 interpreter; see `crates/kopitiam-lua`). The interpreter and its pattern matcher (`pattern.rs`, following the reference manual §5.4.1) are written from the *specification*, not ported from PUC-Rio's `lstrlib.c`/reference implementation. This is a language-specification reference, not a source-code reference; provenance is also named at the point of use. |
| [OpenFOAM](https://github.com/OpenFOAM/OpenFOAM-dev) (OpenFOAM Foundation) | GPL-3.0 | The **finite-volume method** for CFD: unstructured-mesh geometry (`polyMesh`/`fvMesh`), the `fvc`/`fvm` discretization operators, `fvMatrix` assembly, linear solvers, boundary conditions, and `dimensionSet` dimensional analysis. Studied as the reference for building out `kopitiam-discretization`, `kopitiam-equation`, `kopitiam-units`, and future mesh/solver crates. Unlike the permissive references above, OpenFOAM is **GPL-3.0 (copyleft)** — compatible with KOPITIAM's AGPL-3.0-only; any algorithm translated from it retains OpenFOAM attribution at the point of use, not merely here. |

### One narrow exception in the AI runtime, recorded honestly

The AI runtime is clean-room overall, but one small piece does not fit the
"studied, then written from scratch" description and is called out so the
distinction stays honest. `kopitiam-ai`'s ChatML renderer
(`crates/kopitiam-ai/src/local/chat_template.rs`) **transliterates** the trivial
three-line ChatML loop from `llama.cpp`'s `LLM_CHAT_TEMPLATE_CHATML` branch, and
its unit test uses an expected-output string **copied verbatim** from
`llama.cpp`'s own chatml test as a ground-truth oracle. Both are attributed at
the point of use in that file. `llama.cpp` is MIT-licensed, so this reuse is
permitted with its notice retained; the snippet is a short functional template
loop and the copied string is a small test fixture, not an algorithm — but it is
a close adaptation, not clean-room study, and is labelled as such here rather
than being quietly filed under "studied for."

### The default shipped local model — SmolLM2 (downloaded, not vendored)

KOPITIAM's **default local model** is **SmolLM2-360M-Instruct**
(`kopitiam_models::DEFAULT_MODEL_ID`), by **HuggingFaceTB**, under
**Apache-2.0** — tiny (~369 MB Q8_0) and fast enough for local-first / Termux
use, the successor to the earlier Qwen2.5-0.5B default. A larger option,
**SmolLM2-1.7B-Instruct** (Q4_K_M, also Apache-2.0), is offered alongside it.
Both are LLaMA-shaped architectures.

| Project | License | Role |
| --- | --- | --- |
| [SmolLM2](https://huggingface.co/HuggingFaceTB) (HuggingFaceTB) | Apache-2.0 | The **default local model weights**, in GGUF form. Pulled from the official HuggingFaceTB GGUF repos at runtime via `kopitiam-models` (`kopitiam models pull …`, checksum-verified against the catalog's recorded sha256) — **downloaded, never vendored or committed** as weights. Apache-2.0 combines one-way into KOPITIAM's AGPL-3.0-only, and the licence travels with the downloaded files. |

A read-only **reference clone** of the upstream SmolLM repository is vendored
at **`crates/kopitiam-models/vendor/smollm`** (Apache-2.0, HuggingFaceTB),
shallow-cloned for study of the tokenizer and the LLaMA-shaped architecture
(the configs there carry `is_llama_config: true`, which is why the catalog
files these entries under `Architecture::Llama`). Like every other vendored
clone, it is reference material only — never built, linked, or shipped as part
of any KOPITIAM crate — and is gitignored. Note the location differs from the
AI-runtime clones under `crates/kopitiam-ai/vendor/`: this one lives beside the
acquisition layer that knows how to fetch the model.

---

## PDF & document-extraction references (translation / close adaptation)

A second body of reference material sits behind KOPITIAM's PDF stack
(`kopitiam-pdf`, `kopitiam-document`). These libraries are **vendored locally
for study and translation into Rust** — shallow-cloned into gitignored
`crates/*/vendor/` directories — **not** built, linked, or shipped as
dependencies of any KOPITIAM crate. They occupy the middle tier of the model
above: their extraction and layout algorithms are studied and, where a specific
algorithm is ported closely, **translated with attribution at the point of use**
(a doc comment naming the source, its license, and the pinned commit), not only
here. Nothing is copied wholesale; see "Clean-room implementation" below.

Two of them — **PyMuPDF** and its underlying C engine **MuPDF** — are
**AGPL-3.0**. KOPITIAM's relicense to AGPL-3.0-only is precisely what makes
porting algorithms from them permissible: an AGPL upstream can be adapted into
an AGPLv3 work, whereas it could not have been absorbed into a permissive- or
GPL-only-licensed one. The permissively-licensed references (MIT, BSD-3-Clause)
combine one-way into AGPLv3 as usual, with their notices retained.

| Project | License | Pinned commit | Studied / translated for |
| --- | --- | --- | --- |
| [pdfminer.six](https://github.com/pdfminer/pdfminer.six) | MIT | `a18de2a` | PDF char→word→line grouping and layout / reading-order analysis (`kopitiam-pdf` / `kopitiam-document`) |
| [pdfplumber](https://github.com/jsvine/pdfplumber) | MIT | `4c64b92` | Word / table extraction with coordinates |
| [pypdf](https://github.com/py-pdf/pypdf) | BSD-3-Clause | `0a87f78` | Layout-mode text reconstruction cross-check |
| [pdf-to-markdown](https://github.com/iamarunbrahma/pdf-to-markdown) | MIT | `54baa2e` | PDF→Markdown pipeline: multi-column reading order, header / footer stripping, heading / table reconstruction |
| [marker](https://github.com/datalab-to/marker) | Apache-2.0 | — | **Clean-room study only — no code copied.** The running-head / running-foot detection *guard rails* in `kopitiam-document`'s `reconstruction/headers.rs` (the margin-zone constraint and the "be conservative on short documents" minimum-page rule) were informed by reading marker's `processors/marginalia.py` and `processors/ignoretext.py`. The signature-recurrence algorithm itself is adapted from pdf-to-markdown (above); marker was studied for the safety conditions wrapped around it. Original Rust throughout. |
| [PyMuPDF](https://github.com/pymupdf/PyMuPDF) | AGPL-3.0 | — | High-capability text extraction (blocks / lines / spans with bounding boxes); kept purely as an **API-shape** reference (what `get_text` yields). The algorithm KOPITIAM ports is always the underlying C, never the Python binding (AID-0051). Now license-compatible under the AGPLv3 relicense |
| [MuPDF](https://github.com/ArtifexSoftware/mupdf) | AGPL-3.0 | `19f1284` | The C engine PyMuPDF binds — the **primary translation source** for `kopitiam-pdf/src/mupdf/`. The port has grown from text extraction into a near-complete PDF **rendering vertical**: the `fitz` geometry / stream / filter primitives, the PDF object / lexer / parser / xref layer, CMap / font / ToUnicode / AGL, the `stext` structured-text device, **the content interpreter wired to a draw device** (`op_run.rs` / `page_run.rs` / `resources.rs`, from `pdf-op-run.c`), **a pure-Rust rasterizer** (`pixmap.rs` / `draw_edge.rs` / `draw_path.rs` / `draw_device.rs`, from `pixmap.c` / `path.c` / `draw-*.c`), **embedded-font glyph-outline decoding** (`glyph*.rs`), and **embedded-image decode** (`page_image.rs`, from `pdf-image.c` / `image.c` / `load-jpeg.c`), and **annotation rendering** (`annot_run.rs` from `pdf-run.c` / `pdf-annot.c`, `annot_appearance.rs` from `pdf-appearance.c` / `pdf-annot.c`), and — as of 0.2.9 — the **write** path: `write.rs` (from `pdf-write.c`'s incremental-update branch and `pdf-object.c`'s `fmt_obj`/`fmt_str`/`fmt_name` serialisation), `annot_edit.rs` (from `pdf-annot.c`'s `pdf_create_annot`/`pdf_delete_annot`/`pdf_set_annot_*`), and `form.rs` (from `pdf-form.c` + the widget branches of `pdf-appearance.c`). Note the form work deliberately does **not** port `pdf-js.c`: `pdf_set_field_value`'s `ignore_trigger_events` flag makes read/set/toggle/regenerate reachable with no JavaScript engine at all, at the documented cost that script-*computed* fields do not recalculate. Every ported unit carries a point-of-use provenance header naming the source file and pinned commit; `docs/port-ledger.md` enumerates the MuPDF source→target units. **Two pinned commits:** everything up to and including the image path is at `19f1284`; the annotation units are at `5fe54ce`, because the vendored tree is gitignored and so did not survive to the machine that added them — see AID-0056 for why that split was kept and documented rather than back-dated. See AID-0051 (port conventions) and AID-0052 (the C-delegation substitutions below) |
| [tdf](https://github.com/itsjunetime/tdf) | AGPL-3.0 | — | Terminal PDF **viewer** studied for the "tdf-parity" target: how a Rust TUI presents rasterized pages (responsive layout, graphics-protocol detection, half-block fallback) that the new draw-device / `ratatui-image` viewer path matches. Clean-room study of the UX only; the rasterizer itself is the MuPDF translation above |

As with every other reference here, these are credited up front, their upstream
licenses are recorded and honored, and any close translation records its
provenance at the point of use — they are studied, not shipped.

### Pure-Rust substitutions for the C libraries MuPDF delegates to

MuPDF is not self-contained: for three jobs it hands off to a bundled C library
(`<zlib.h>` for FlateDecode, `<jpeglib.h>` for DCTDecode, **FreeType** for
embedded-font outlines). KOPITIAM's Pure Rust Core forbids a mandatory C
toolchain, so the port does **not** translate those C libraries. Instead, where
MuPDF would call one, the port either **substitutes a pure-Rust crate** that is
linked and shipped, or **re-implements the format from its public specification**
(clean-room). This is a distinct provenance relationship from the MuPDF
translation above, and is recorded here and in **AID-0052**:

* **FlateDecode / zlib → `miniz_oxide`.** `filter_flate.rs` wires miniz_oxide's
  streaming inflate into an `fz_stream`-style filter in place of MuPDF's zlib
  call. The filter *plumbing* is translated from MuPDF; the DEFLATE codec is the
  pure-Rust crate.
* **DCTDecode / libjpeg → `zune-jpeg`.** `page_image.rs` translates MuPDF's
  `/Width /Height /ColorSpace /Decode /Filter` reading and sample unpacking
  (`pdf-image.c` / `image.c`), but the JPEG bitstream itself is decoded by the
  pure-Rust `zune-jpeg` (with `zune-core`) rather than by `load-jpeg.c`'s
  libjpeg. `MIT OR Apache-2.0 OR Zlib`; no C in the build.
* **Embedded-font outlines / FreeType → spec re-implementation, plus a
  skrifa second opinion.** MuPDF reads `/FontFile2` / `/FontFile3` /
  `/FontFile` programs through FreeType. This port **avoids FreeType
  entirely**: `glyph.rs` keeps MuPDF's outline→`Path` *callback shape* (the
  `move_to` / `line_to` / `conic_to` / `cubic_to` decompose of `font.c`), but
  the actual font-program parsing is written **clean-room from the format
  specifications** — the OpenType `glyf`/`loca` tables (`glyph_truetype.rs`),
  the Adobe CFF / Type2 Charstring format, Technical Note #5177
  (`glyph_cff.rs`), and the **Adobe Type 1 Font Format specification** (Adobe
  Systems Inc., 1990 — `glyph_type1.rs`, the `/FontFile` decoder: PFA/PFB
  unwrapping, `eexec`/charstring decryption, the Type1 charstring interpreter
  including `seac` and the `OtherSubrs` flex/hint-replacement convention). One
  narrow exception is recorded at the point of use: the CFF *container* parse
  (`INDEX` / `DICT` / charset / FDSelect / `subr_bias`) is a close adaptation
  of MuPDF's own non-FreeType CFF reader in `source/fitz/subset-cff.c` (still
  MuPDF, still `19f1284`), not of FreeType. The CFF **Standard Strings** table
  (`glyph_cff.rs`, `CFF_STANDARD_STRINGS`, Adobe TN#5176 Appendix A — used to
  resolve a predefined-Standard-encoding simple CFF font's `code -> name ->
  gid`) is the spec's own fixed 391-entry list, not creative expression to
  attribute; it was cross-checked against fontTools'
  `cffLib.cffStandardStrings` (`fonttools`, BSD-3-Clause) for transcription
  accuracy rather than typed from memory. See **AID-0055**.

  As of the skrifa integration (`glyph_skrifa.rs`), the clean-room decoders
  above remain the **primary** path and are tried first for every glyph;
  [skrifa](https://crates.io/crates/skrifa) (Google `fontations`,
  Apache-2.0/MIT) is consulted **only** as a per-glyph second opinion when a
  primary decoder returns no outline for a specific GID (the documented
  ceilings: predefined-Expert CFF encoding, CID-keyed CFF edge cases, `seac`).
  This is a genuine crates.io dependency (see "Notable shipped Rust
  dependencies" below), not a translation — it is used as published, unmodified.

The consequence for provenance is that these subsystems are **not** derivative
of FreeType or libjpeg: the glyph decoders are spec-based original Rust (plus one
MuPDF-`subset-cff.c` adaptation), and the JPEG/zlib paths are permissive shipped
crates whose notices travel with the binary. See "Notable shipped Rust
dependencies" below.

## OCR references (translation / close adaptation)

KOPITIAM's OCR fallback (`kopitiam-ocr`) — the automatic path when a PDF has no
embedded text — is a **pure-Rust translation of the Tesseract LSTM engine**
(English, Simplified Chinese, Japanese), running its recognizer on
`kopitiam-tensor` and fetching `.traineddata` language models through
`kopitiam-models`. Both upstreams are permissively licensed and translate cleanly
into AGPL-3.0; every ported file keeps the upstream copyright / author / license
header at the point of use, and the language models are downloaded, never
vendored or committed.

| Project | License | Pinned commit | Studied / translated for |
| --- | --- | --- | --- |
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) | Apache-2.0 | `db0ec62` | The LSTM OCR engine: `.traineddata` container + unicharset/recoder parsers, the LSTM recognizer and CTC/recode beam decoder, line-finding (`kopitiam-ocr`) |
| [Leptonica](https://github.com/DanBloomberg/leptonica) | BSD-2-Clause | `10bdea2` | Image preprocessing on the OCR path: binarization (Otsu / Sauvola), grayscale, scale-to-line-height (`kopitiam-ocr`) |
| [tessdata_best](https://github.com/tesseract-ocr/tessdata_best) | Apache-2.0 | — | Source of the `eng` / `chi_sim` / `jpn` LSTM language models — **downloaded at runtime via `kopitiam-models`, not vendored** |

---

## kovan (sibling project — reader features ported across)

`kovan` is the maintainer's own literature/digitiser workbench, living in
[`theodoreOnzGit/outram-park-backend`](https://github.com/theodoreOnzGit/outram-park-backend)
(`crates/kovan`, branch `develop`). It is not a third-party dependency and
nothing is vendored from it: KOPITIAM and kovan share an author, and `kpdf`
carries features **ported** from kovan's egui PDF reader
(`src/digitiser/gui/desktop/pdf_reader.rs`) so that PDF work consolidates in
`kopitiam-pdf` rather than being maintained twice.

| Ported into | From kovan | Notes |
| --- | --- | --- |
| `gui_frontend/hot_reload.rs`, `kpdf`'s `check_hot_reload` | `PdfReaderState::check_hot_reload` / `read_mtime`, `RELOAD_CHECK_INTERVAL` (kovan issue #30, op-eehc) | Same mechanism (throttled mtime poll, not a filesystem watcher), same 500 ms interval, same default-on stance. Diverges deliberately: the clock is injected so the throttle is unit-testable, and kpdf additionally refuses to auto-reload over **unsaved annotation edits** — a case kovan's reader does not have to handle, and silently discarding a user's ink would be data loss. |
| continuous scroll, thumbnail sidebar, page-LRU (earlier work) | kovan's reader | See `kpdf`'s module docs for what was and was not carried over. |

Deliberately **not** ported: kovan's box-annotation / graph-digitiser /
table-reading tooling and its 3-pane bibliography panels — a different
workflow from kpdf's ink/eraser/forms tools. Where the two overlapped,
whichever implementation was more advanced won: kovan's text extraction and
image extraction go through `pdf-extract`/`lopdf`, which `kopitiam-pdf`'s own
MuPDF port already supersedes, so those were **not** taken.

---

## Forks (direct code reuse — notices retained)

Unlike everything above, these are **forks**. Their code is reused directly,
their copyright notices are retained, and the crate's rustdoc says so.

| Project | License | Fork commit | KOPITIAM crate | Why forked |
| --- | --- | --- | --- | --- |
| [rmux](https://github.com/helvesec/rmux) | MIT OR Apache-2.0 | — | `kmux` (binary `kmux`) | Terminal multiplexer, already Rust, but it does not run on Android. Forked to add Android/Termux support alongside Linux, macOS and Windows. Now **collapsed into a single publishable `kmux` crate** (see below). Upstream copyright: **"The RMUX Authors"**. |
| [beads-rs](https://github.com/delightful-ai/beads-rs) | MIT | `d98da23` | `kopi-beans` (binary `bn`) | The `bd`-style issue tracker KOPITIAM uses for its own work items (every AID is filed as an issue), already Rust, but upstream v0.1.26 / 0.2.0-alpha does not build on Windows (~19 errors: Unix-only deps and a Unix-domain-socket-only daemon IPC layer). Forked to add Windows and Android/Termux support, then **collapsed into a single publishable `kopi-beans` crate and made fully pure-Rust** (see below). Binary renamed `bd` → `bn`. Upstream copyright: **© 2025 Darin Kishore**. |

### What the rmux fork actually consists of

Recorded plainly, because "fork" is doing a lot of work in that table row.

**The whole of rmux was taken**, not a subset — roughly 325k lines. It was
originally forked as its twelve upstream crates nested under
`crates/kmux/crates/`, keeping their upstream names so that diffs against
upstream stayed readable. **The overwhelming majority of the code in `kmux` was
written by The RMUX Authors, not by KOPITIAM.**

**Single-crate collapse (for crates.io).** To publish `kmux` to crates.io
without squatting upstream rmux's sub-crate names, the ten sub-crates the `kmux`
binary actually needs (`rmux-core`, `rmux-os`, `rmux-pty`, `rmux-ipc`,
`rmux-proto`, `rmux-types`, `rmux-client`, `rmux-server`, `rmux-sdk`,
`rmux-web-crypto`) were folded into **one self-contained `kmux` crate** as the
intra-crate modules `core` / `os` / `pty` / `ipc` / `proto` / `types` / `client`
/ `server` / `sdk` / `web_crypto`. `ratatui-rmux` and `rmux-render-core`, which
the binary never linked, were dropped. Cross-crate paths (`rmux_core::X`) became
`crate::core::X`; the sub-crate manifests were unioned into one; the former
`rmux-server` build script (tunnel-preset codegen) merged into this crate's
`build.rs`. Upstream dependency pins were carried verbatim. The reasoning behind
the collapse is recorded in **AID-0053**.

* Upstream's `LICENSE-MIT` and `LICENSE-APACHE` ship unmodified in
  `crates/kmux/`, and `crates/kmux/NOTICE` records the fork **and the collapse**.
* The folded modules' rustdoc names The RMUX Authors and the original license.
* The fork is distributed under **AGPL-3.0-only** as part of KOPITIAM, which the
  permissive upstream licenses allow so long as their notices travel with the
  code. **This does not relicense rmux**, which remains available from its
  authors under MIT OR Apache-2.0.
* Upstream's release/packaging/CI scripts, benchmarks, `xtask` and contributor
  documentation were *not* carried into the fork.

KOPITIAM's own contributions to the fork are small and concentrated: the
Android/Termux port (`cfg` widening, the `rmux_os::runtime_dir` resolver, the
Bionic-specific PTY/signal/locale paths) and the `kmux` binary rename. See
`docs/ai-decisions/AID-0006`.

### What the beads-rs fork actually consists of

The same "fork" caveats apply to `kopi-beans` (binary `bn`), and are recorded in
full in `crates/kopi-beans/NOTICE`.

**The whole of beads-rs was taken**, not a subset: the top package plus every
nested sub-crate (beads-core, beads-api, beads-bootstrap, beads-surface,
beads-macros, beads-cli, beads-git, beads-daemon, beads-daemon-core), forked from
upstream `main` at commit `d98da231` (post-dates the tagged v0.1.26; reports
itself as `0.2.0-alpha`). **The overwhelming majority of this code was written by
the beads-rs author, Darin Kishore, not by KOPITIAM.**

**Single-crate collapse (for crates.io).** The fork began as the former
multi-crate `kopitiam-bds` (top package plus nine path-dependency sub-crates).
To publish to crates.io without squatting upstream's crate names, those
sub-crates were folded into **one self-contained `kopi-beans` crate** as the
intra-crate modules `core` / `api` / `bootstrap` / `surface` / `macros` /
`cli_surface` / `git` / `daemon` / `daemon_core`, and the binary was renamed
`bd` → `bn`. The library keeps the upstream lib name `beads_rs`
(`[lib] name = "beads_rs"`) so folded source and future upstream merges stay
undisturbed, while the published **package** name `kopi-beans` squats none of
upstream's crate names. `beads-macros` is *declarative* macros (not a proc-macro),
so it folds in as an ordinary module. Not carried into the collapsed crate: the
`tests/` integration suites, the dev-only `beads-http` (axum/tokio) transport,
the `fuzz` crate, and the `beads_stateright_models` model-checking harness;
inline `#[cfg(test)]` unit tests within the folded modules **are** carried. The
collapse convention is recorded in **AID-0053**.

**Now fully pure-Rust and C-free** — which is what makes it cleanly publishable.
The original fork still pulled two bundled-C dependencies (`rusqlite`'s bundled
SQLite, `git2`'s libgit2); both were removed:

* **`rusqlite` (bundled C SQLite) → a pure-Rust `MemoryWalIndex`.** The index
  layer is now pure Rust with no libsqlite3.
* **`git2` / libgit2 → `gitoxide` (`gix`).** The git integration is ported to the
  pure-Rust `gix` (MIT OR Apache-2.0). The tree now contains none of `git2` /
  `libgit2-sys` / `openssl-sys` / `libz-sys`, so `bn` cross-compiles to
  Termux/Android with no NDK. **One gap is gated, not hidden:** `gix` 0.86
  exposes no high-level push, so `bn`'s push path is gated pending a
  `gix-protocol` / `gix-transport` send-pack shim; fetch and `file://` /
  `git://` / `ssh` round-trips work. See "Notable shipped Rust dependencies"
  below for the `gix` credit.

* Upstream's `LICENSE-MIT` ships unmodified in `crates/kopi-beans/`, and
  `crates/kopi-beans/NOTICE` records the fork, the commit, the collapse, and
  every change.
* The fork is distributed under **AGPL-3.0-only** as part of KOPITIAM, which the
  permissive upstream MIT license allows so long as its notice travels with the
  code. **This does not relicense beads-rs**, which remains available from its
  author under MIT.
* KOPITIAM's contributions are concentrated in **Windows support** (the reason
  for the fork: a cross-platform Unix-domain-socket alias, Win32 process
  liveness / `LockFileEx` store locking / `CreateProcessW` daemon spawn, `cfg`
  gating of POSIX signals and `nix`/`libc`/`signal-hook`), the Android/Termux
  code path (`cfg(unix)` covers `target_os = "android"`), and the pure-Rust
  `rusqlite`/`git2` removals above. Upstream's release/packaging/CI scripts, the
  `fuzz` crate, and the `stateright` model-checking harness were not carried into
  the fork's build.

---

## Notable shipped Rust dependencies (linked, notices retained)

Everything above is either studied (never linked), or a whole-project fork. This
section names the smaller set of **ordinary crates.io dependencies** that KOPITIAM
links and ships and that are worth crediting individually — because they either
**substitute for a C library** an upstream used (keeping the Pure Rust Core), or
are load-bearing enough to name. These are used **as published**, unmodified;
Cargo carries their license text, and their permissive licenses combine one-way
into AGPLv3. (Routine crates — serde, tokio, ratatui, clap, … — are not
enumerated here; their provenance is the Cargo lockfile.)

| Crate | License | Role |
| --- | --- | --- |
| [pdf-extract](https://crates.io/crates/pdf-extract) | MIT | The pre-MuPDF-port PDF text-extraction path (wraps `lopdf`); still the `pdf-extract` engine option in `pdf2md` |
| [lopdf](https://crates.io/crates/lopdf) | MIT | Low-level PDF object / content-stream walking for font-style recovery (`kopitiam-pdf`), `kopitiam-plot` vector paths |
| [zune-jpeg](https://crates.io/crates/zune-jpeg) (+ `zune-core`) | MIT OR Apache-2.0 OR Zlib | Pure-Rust JPEG decoder substituting for MuPDF's libjpeg on the DCTDecode image path (see "Pure-Rust substitutions" above, AID-0052) |
| [skrifa](https://crates.io/crates/skrifa) (Google `fontations`) | MIT OR Apache-2.0 | Per-glyph **second opinion** for embedded-font outlines (`glyph_skrifa.rs`): consulted only when the clean-room `glyph_truetype.rs`/`glyph_cff.rs` decoders return no outline for a specific GID. The primary decoders stay primary and unmodified; see "Pure-Rust substitutions" above |
| [miniz_oxide](https://crates.io/crates/miniz_oxide) | MIT OR Zlib OR Apache-2.0 | Pure-Rust DEFLATE/zlib behind PDF FlateDecode, substituting for MuPDF's zlib |
| [gitoxide](https://crates.io/crates/gix) (`gix`) | MIT OR Apache-2.0 | Pure-Rust Git, substituting for `git2`/libgit2 in the `kopi-beans` fork — no libgit2-sys / openssl-sys / libz-sys, so `bn` cross-compiles to Termux/Android with no NDK. `gix` 0.86 has no high-level push, so the push path is gated on a `gix-protocol`/`gix-transport` send-pack shim (see the beads-rs fork section) |
| [nucleo](https://crates.io/crates/nucleo) / [nucleo-matcher](https://crates.io/crates/nucleo-matcher) | MPL-2.0 | Fuzzy matching/ranking by the Helix authors — kvim's telescope-replacement pickers and the CLI/TUI PDF finder. MPL-2.0 is file-level copyleft, one-way compatible with AGPLv3; used unmodified as a dependency, so its files stay under MPL-2.0 |
| [ratatui-image](https://crates.io/crates/ratatui-image) | MIT | The PDF viewer's image mode: graphics-protocol detection (kitty / sixel / iTerm2) with a Unicode half-block fallback for Termux. Taken with only the `crossterm` feature — the `chafa` C-linking features are deliberately left off |
| [portable-pty](https://crates.io/crates/portable-pty) / [vt100](https://crates.io/crates/vt100) | MIT | kvim's `:term` emulator — pty spawn + ANSI/VT parsing, both pure Rust (see AID-0049) |

---

## Bundled assets

| Asset | License | Why bundled |
| --- | --- | --- |
| JetBrains Mono Nerd Font Mono (Regular) | OFL-1.1 (font); Nerd Fonts patcher is MIT | Shipped **inside** `kopitiam-neovim` so `kvim` renders devicons on Android, whose terminals have no Nerd Font. A devicon is a Private-Use-Area codepoint, so shipping the icon table alone would render tofu boxes — the font itself has to travel with the binary. See `docs/ai-decisions/AID-0004`. OFL governs the font as a distinct work: it does not infect the AGPLv3 program that bundles it, but its copyright and license text must travel with it, and it may not be sold on its own. Both conditions are honoured. |

---

## License compatibility with AGPLv3

KOPITIAM is licensed AGPL-3.0-only (see README.md's "Why AGPLv3,
specifically?"). Every permissively-licensed project above (MIT,
Apache-2.0, BSD-2/3-Clause, Zlib) is one-way compatible with AGPLv3: permissive
code can be incorporated into an AGPLv3 work, provided its copyright
notices and license text are retained, and the combined work is then
distributed under AGPLv3 as a whole. The **copyleft** upstreams combine too:
the AGPL-3.0 ones (MuPDF, PyMuPDF, tdf) are the very reason the relicense to
AGPL-3.0-only was made, GPL-3.0 (OpenFOAM) is upgrade-compatible with AGPLv3,
and MPL-2.0 (nucleo) is file-level copyleft that co-exists with an AGPLv3 work
so long as the MPL files keep their license — which, used unmodified as a
dependency, they do.

None of this is a license to copy code wholesale. See "Clean-room
implementation" below.

## Clean-room implementation

KOPITIAM's Translation Philosophy (`CLAUDE.md`) already states this for
legacy-language translation; it applies equally here:

1. Read the papers, documentation, and public APIs of the reference project.
2. Understand the algorithm, not just the code.
3. Design a Rust-native abstraction for it.
4. Write original Rust code implementing that abstraction.
5. Validate against the reference implementation with benchmarks and tests.

Do not translate any of the above repositories line-by-line. If a specific
function or algorithm is adapted closely enough from one of them that
attribution is warranted beyond this file (e.g. a specific quantization
kernel or sampling algorithm), record that provenance in the Rust source
itself — a doc comment naming the source and its license — not only here.

## Where the clones live

Local, read-only reference clones of the projects above live under
`crates/kopitiam-ai/vendor/`, shallow-cloned (`--depth 1`, no history) and
excluded from version control by `.gitignore`. They exist for the
implementer (human or AI) to read while building the runtime described in
the parent epic tracked by `kopitiam-082`; nothing under `vendor/` is a
build dependency of any KOPITIAM crate.

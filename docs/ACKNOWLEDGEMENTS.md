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
| [Helix](https://github.com/helix-editor/helix) | MPL-2.0 | Modal-editor **infrastructure and feature-completeness reference** for `kvim` — how a mature Rust editor wires LSP lifecycle, incremental syntax, a command palette, and buffer/window management. **kvim is vim-modeled**, so Helix's selection-first keymap is studied for *what* mature editors do, never for *how* kvim binds keys. Clean-room study only: no Helix code is copied, and MPL-2.0 governs any file that ever were — none is. |
| [Language Server Protocol specification](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/) | CC-BY-4.0 (spec text) | The **snippet syntax grammar** implemented clean-room by `kopitiam-snippet` (`$1`, `${1:…}`, `${1\|a,b\|}`, `$VAR`, escapes, mirrors). Only the published *grammar* is followed; **no code is copied** from LuaSnip, vsnip, or VS Code. This is a specification reference, not a source-code reference. |
| [Lua](https://www.lua.org/) 5.1 (PUC-Rio) — the [Lua 5.1 Reference Manual](https://www.lua.org/manual/5.1/) | MIT | The **Lua 5.1 language** implemented clean-room by `kopitiam-lua` (`kopitiam-lua` is a pure-Rust Lua 5.1 interpreter; see `crates/kopitiam-lua`). The interpreter and its pattern matcher (`pattern.rs`, following the reference manual §5.4.1) are written from the *specification*, not ported from PUC-Rio's `lstrlib.c`/reference implementation. This is a language-specification reference, not a source-code reference; provenance is also named at the point of use. |

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

---

## Forks (direct code reuse — notices retained)

Unlike everything above, these are **forks**. Their code is reused directly,
their copyright notices are retained, and the crate's rustdoc says so.

| Project | License | KOPITIAM crate | Why forked |
| --- | --- | --- | --- |
| [rmux](https://github.com/helvesec/rmux) | MIT OR Apache-2.0 | `kopitiam-mux` (`kmux`) | Terminal multiplexer, already Rust, but it does not run on Android. Forked to add Android/Termux support alongside Linux, macOS and Windows. Upstream copyright: **"The RMUX Authors"**. |

### What the rmux fork actually consists of

Recorded plainly, because "fork" is doing a lot of work in that table row.

**The whole of rmux was taken**, not a subset: all twelve of its crates
(`rmux-core`, `rmux-os`, `rmux-pty`, `rmux-ipc`, `rmux-proto`, `rmux-types`,
`rmux-client`, `rmux-server`, `rmux-sdk`, `rmux-render-core`, `rmux-web-crypto`,
`ratatui-rmux`) plus its top-level binary — roughly 325k lines. They live under
`crates/kopitiam-mux/crates/`, keeping their upstream names so that diffs
against upstream remain readable. **The overwhelming majority of the code in
`kopitiam-mux` was written by The RMUX Authors, not by KOPITIAM.**

* Upstream's `LICENSE-MIT` and `LICENSE-APACHE` ship unmodified in
  `crates/kopitiam-mux/`, and `crates/kopitiam-mux/NOTICE` records the fork.
* Every forked sub-crate's rustdoc names The RMUX Authors and its original
  license.
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

---

## Bundled assets

| Asset | License | Why bundled |
| --- | --- | --- |
| JetBrains Mono Nerd Font Mono (Regular) | OFL-1.1 (font); Nerd Fonts patcher is MIT | Shipped **inside** `kopitiam-neovim` so `kvim` renders devicons on Android, whose terminals have no Nerd Font. A devicon is a Private-Use-Area codepoint, so shipping the icon table alone would render tofu boxes — the font itself has to travel with the binary. See `docs/ai-decisions/AID-0004`. OFL governs the font as a distinct work: it does not infect the AGPLv3 program that bundles it, but its copyright and license text must travel with it, and it may not be sold on its own. Both conditions are honoured. |

---

## License compatibility with AGPLv3

KOPITIAM is licensed AGPL-3.0-only (see README.md's "Why AGPLv3,
specifically?"). Every permissively-licensed project above (MIT,
Apache-2.0, BSD-3-Clause) is one-way compatible with AGPLv3: permissive
code can be incorporated into an AGPLv3 work, provided its copyright
notices and license text are retained, and the combined work is then
distributed under AGPLv3 as a whole.

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

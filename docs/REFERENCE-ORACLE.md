# The llama.cpp reference oracle

How to check KOPITIAM's inference against a known-good implementation, on the
**same weights** with the **same prompt**. Built during the 2026-07-27/28
session that produced `bd-b9x`, and written down because deriving it again costs
hours.

This is the CLAUDE.md "vendored references come FIRST" rule at its most literal:
when the question is *"is our forward pass right?"*, no amount of reading our own
code answers it. Running both and diffing does.

## Build it (once)

`crates/kopitiam-ai/vendor/llama.cpp` is a gitignored reference clone. On the
maintainer's Windows box (cmake 3.29, Ninja, MSVC 14.44 all already present):

```bash
cd crates/kopitiam-ai/vendor/llama.cpp
cmake -B build -G "Visual Studio 17 2022" -A x64 \
      -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON
cmake --build build --config Release --target llama-completion llama-tokenize -j 8
```

Two gotchas that cost time:

* **`llama-cli` is not the target you want.** It now depends on the server, so it
  does not build with `-DLLAMA_BUILD_SERVER=OFF`. `llama-completion` is the raw
  prompt-in/completion-out tool and is what you want for comparison anyway.
* **`-f file` strips the file's trailing newline; `-e -p '...'` does not.** That
  is a one-token difference, which silently makes the comparison not
  apples-to-apples. Always use `-e -p` when the exact token count matters.
  (`llama-tokenize -f` does *not* strip it — the two tools disagree.)

## Compare an output

Greedy on both sides, so the comparison is deterministic and no sampler
difference can be blamed:

```bash
LC=crates/kopitiam-ai/vendor/llama.cpp/build/bin/Release/llama-completion.exe
M=~/AppData/Local/kopitiam/models/smollm2-360m-instruct-q8_0/smollm2-360m-instruct-q8_0.gguf

"$LC" -m "$M" -e -p 'The capital of France is' -n 5 --temp 0 --seed 0 -no-cnv --no-warmup
```

Ours, greedy, via `kopitiam_runtime::generate` with a default `GenerationConfig`.
The netfetch harness already does this for one canonical sentence — see
`canonical_completion` in `crates/kopitiam-runtime/tests/netfetch_end_to_end.rs`.

## Compare the tokenisation

Divergence is worth ruling out at the tokenizer before blaming the model:

```bash
LT=crates/kopitiam-ai/vendor/llama.cpp/build/bin/Release/llama-tokenize.exe
printf 'The capital' > /tmp/t.txt
"$LT" -m "$M" -f /tmp/t.txt --ids     # -> [504, 3575]
"$LT" -m "$M" -f /tmp/t.txt           # id -> token text
```

Ours: `kopitiam_runtime::tokenizer_from_gguf(&model)?.encode(text)?`.

As of this writing the two agree byte for byte on the full ChatML prompt,
including splitting `assistant` into `"ass"` + `"istant"` — so a tokenisation
difference is **not** the explanation for `bd-b9x`.

## Remove the variables before concluding anything

Each of these silently breaks a comparison, and each cost real time:

| Variable | How to control it |
|---|---|
| Sampling | `--temp 0` on llama.cpp; `generate` (greedy) on ours. |
| KV-cache precision | llama.cpp defaults to **f16**; force parity with `-ctk f32 -ctv f32`. (Made no difference to the cases tested.) |
| BOS | This model sets `add_bos_token=false` and llama.cpp honours it — verify with `llama-tokenize --ids`, do not assume. |
| Trailing newline | See the `-f` vs `-e -p` gotcha above. |
| Our quantized kernel | `KOPITIAM_FORCE_F32_WEIGHTS=1` dequantizes every matmul weight, so a quantized-kernel bug can be told apart from a forward-pass bug. |

## What "agreement" looks like

On a prompt the model is confident about, agreement is **exact**, and that is the
bar to hold:

```text
"The capital of France is"        ours " Paris"=20.337 rank 0   llama.cpp " Paris"
"The quick brown fox ... lazy"    ours " dog"=22.597  rank 0    llama.cpp " dog"
```

Where the model is genuinely uncertain (top candidates within ~0.5 logits), the
argmax can legitimately flip on floating-point differences. Judge those by the
**margin**, not by whether the strings match. A pick that sits 3+ logits down our
ranking — as in `bd-b9x`, where llama.cpp's choice is our rank 69 — is not a
tie-break, it is a disagreement.

## Using this for the GPU path

Any wgpu offload must be compared **twice**: against our own CPU kernels (which
are individually verified exact against hand-computed references) and against
llama.cpp end to end. Note the ordering constraint: while `bd-b9x` is open our
CPU forward pass itself disagrees with llama.cpp, so a GPU-vs-llama.cpp
comparison would inherit that error and a genuine GPU bug could not be
distinguished from the existing one. Fix `bd-b9x` first, then a GPU/CPU/llama.cpp
three-way comparison is meaningful.

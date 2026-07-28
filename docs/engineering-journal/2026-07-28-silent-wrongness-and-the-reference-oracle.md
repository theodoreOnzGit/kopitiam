# Silent wrongness, and why only a reference oracle finds it

**Date:** 2026-07-28
**Related:** `bd-b9x` (RoPE), `bd-8f8` (tied embedding), `bd-11b`/`bd-ztz` (the
hunt), AID-0054, `docs/REFERENCE-ORACLE.md`, `crates/kopitiam-runtime/src/rope.rs`

A ~14-hour session that started at "SmolLM2 won't load" and ended three real
bugs later. Every one of them was **silently wrong** — the code ran, produced
plausible output, and passed its own tests. Writing down how they were actually
found, because the method transfers and the specific hours do not.

---

## The three bugs, shortest form

| Bug | Symptom | Why every test passed |
|---|---|---|
| Byte-level vocab check too strict (AID-0054) | Default model refused to load at all | The only loud one. Synthetic fixtures were built the way the code expects, not the way HuggingFace ships |
| **Wrong RoPE convention** (`bd-b9x`) | Fluent, confidently wrong answers; degraded with prompt length | At position 0 both conventions are the identity. Both preserve vector norm, so the output always *looks* like a valid rotation |
| Tied embedding dequantized (`bd-8f8`) | Nothing. Just 4x memory and 4x slower | Correct output. Only a profile or a memory count reveals it |

---

## The thing that actually mattered: a reference oracle

Every component of the forward pass had been individually verified **correct**
against hand-computed references — RoPE numerically exact at positions 0..100,
mask broadcast across all heads, `q@Kᵀ` and `probs@V` at real shapes, softmax,
`rms_norm` with a non-uniform weight, `silu`, `gather_rows`, Q4_K/Q6_K matching
`ggml-quants.c` byte-for-byte including the signed-`i8` trap. All green. The
composed model still produced garbage.

**Reading our own code could not have found it, and did not, for hours.** What
found it in about twenty minutes was building `llama.cpp` from the vendored
clone and diffing intermediate tensors layer by layer:

```text
              ours(before)   ours(after)   llama.cpp
inp_embd        -6.247214     -6.247214    -6.247214   <- always matched
attn_norm-0     -4.775917     -4.775917    -4.775916   <- always matched
Qcur-0.rope    -32.685047    -37.304260   -37.306450   <- divergence starts HERE
attn_out-0     +11.727785    +11.524756   +11.535707
```

Two lines of numbers localised what a day of reasoning could not. The procedure
is in `docs/REFERENCE-ORACLE.md` so nobody pays that cost again.

**Rule of thumb won here:** when the question is *"is our implementation
right?"*, the answer never comes from reading the implementation. Get a second
implementation and diff. This is what CLAUDE.md's "vendored references come
FIRST" rule is actually protecting.

---

## Why RoPE hid so well, and the general shape of it

`llama.cpp`'s `llama_model_rope_type` puts `LLM_ARCH_LLAMA` in the
**NORM/interleaved** group and `LLM_ARCH_QWEN2` in the **NEOX/split-half** group.
Our `rope.rs` implemented split-half exclusively and its docs asserted "LLaMA and
every model derived from it — including every Qwen release — uses split-half."
Half right, and the wrong half was the default model.

The trap underneath: **HuggingFace's `LlamaAttention` really does use
`rotate_half` (split-half) in Python.** `convert_hf_to_gguf.py` permutes the Q/K
rows so ggml's interleaved rotation reproduces it. So reading the HF modelling
code and concluding "split-half" is *correct about the architecture and wrong
about the file*. A reference is only a reference for the artefact you actually
load.

Three properties made it invisible, and they generalise to a whole class:

1. **Identity at the origin.** Position 0 has angle 0, so both conventions agree
   on the first token. Anything validated with a short prompt passes.
2. **Norm-preserving.** Both are rotations, so shape checks, finiteness checks
   and "does it look like a plausible tensor" all pass.
3. **Graceful degradation.** Error grows with position, so it reads as *"the
   model is small and weak"* — an explanation that is always available and
   always comforting.

If a bug has all three, no amount of internal testing will find it.

---

## A test can assert the bug

`llama_and_qwen2_arch_with_identical_weights_produce_bit_identical_logits`
demanded the two architectures agree bit-for-bit, reasoning that "the arch string
is only a metadata-routing key; nothing in the forward pass branches on it."

That reasoning **was the bug**, and the test passed happily for exactly as long
as the bug existed — because the bug is what made the two paths identical. It now
asserts the opposite.

**An equality test is only as good as the argument that the two things should be
equal.** Worth re-reading any test whose body is "these must match" and asking
where that claim came from.

---

## How I got it wrong, twice, and what fixed the method

Both wrong turns came from the same error: **concluding from output that looked
sensible instead of output that matched a reference.**

* Claimed the cause was the pre-tokenizer (`tokenizer.ggml.pre = "smollm"`).
  Refuted: our token ids are byte-identical to `llama-tokenize`, including the
  `"ass"+"istant"` split.
* Then claimed the runtime was fine and the model was just weak — because an
  18-token completion came out coherent. Also wrong; it simply didn't match the
  reference, which I hadn't checked yet.

There was also a self-inflicted measurement error worth naming: a length ladder
built from **one sentence repeated five times**, which is precisely the pattern
that triggers induction-head copying. It measured repetition and I read it as
length. When a probe produces a striking result, check the probe before
believing it.

---

## Optimising against a broken baseline flatters the wrong thing

The GPU output-head offload measured **4.9x**. Then `bd-8f8` fixed the CPU path
(the tied embedding was being dequantized to f32, skipping the fused quantized
kernel), and the same GPU work re-measured at **~6%**.

Nothing about the GPU code changed. The baseline had been defective, so every
comparison against it was generous.

**Before optimising, verify the baseline is correct.** The corollary bit here in
a useful direction too: the CPU fix was worth 3.8–4.9x and 138 MB, far more than
the GPU work, and it helps precisely where there is no GPU (Termux).

---

## Measure before wiring, not after

The instinct on "offload some tensor ops to wgpu" is to offload them all. The
measurements said otherwise:

```text
decode  attn q/o     (1 tok)   cpu   443µs   gpu 1.88ms   0.24x   LOSS
prefill attn q/o    (33 tok)   cpu 12.89ms   gpu 3.72ms   3.46x   WIN
```

A decode step is **one row of activations** — nowhere near enough arithmetic to
amortise moving a weight matrix. Uniform offload would have made chat 2–4x
*slower* than doing nothing. The win needs either many rows (prefill) or a
resident weight.

Benchmarking also found a bug review had not: wgpu treats an over-limit storage
binding as a validation error and **panics**, aborting the process rather than
returning `Err` for the cascade to catch — and SmolLM2's output head (188 MB f32
vs a 128 MB limit) triggers it exactly.

---

## Small operational lessons, dearly bought

* **`cargo test --workspace` stops at the first failing crate.** Everything
  alphabetically after `kmux` silently never ran. Use `--no-fail-fast`.
* **Never pipe `cargo` through `Select-Object -First N` in PowerShell.** It
  closes the pipe and *kills cargo mid-run*, which looks exactly like a short
  passing run. Cost two bogus results.
* **llama.cpp's `-f file` strips the trailing newline; `-e -p '...'` does not,
  and `llama-tokenize -f` does not either.** A one-token difference between two
  tools in the same repo, enough to invalidate a comparison.
* **`vendor/` is not present on a fresh clone** (gitignored shallow clones), so
  tests that read it fail everywhere, not just in containers. An earlier note
  claiming otherwise sent me looking in the wrong place.
* **A status line that is confidently wrong is worse than none.** The CLI printed
  "Using local model on CPU" long after the output projection had moved to the
  GPU.

---

## The habit worth keeping

Every one of these bugs was found by **comparing against something external** —
llama.cpp's tensors, ggml's `rotate_pairs`, `llama-tokenize`'s ids, a wall-clock
measurement — and none by inspection, however careful.

So: when something is "wrong but runs", stop reading and go get an oracle.
Vendor the upstream if it isn't already there; building `llama-completion` took
minutes and repaid it many times over.

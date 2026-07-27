# AID-0054: byte-level BPE **tolerates holes** in the 256-byte alphabet instead of refusing the model

Status: Pending review
Date: 2026-07-27
Crate: `kopitiam-tokenizer` (`src/bpe.rs`), `apps/cli` (`src/vocab_inspect.rs`),
`crates/kopitiam-runtime/tests/netfetch_end_to_end.rs`
Related: commit `896f5c9` (`kopitiam models inspect`, built precisely to settle
this), commit `1b30375` (the netfetch harness that proved it end to end)

## Context

SmolLM2-360M-Instruct is KOPITIAM's **default local model** (`DEFAULT_MODEL_ID`).
It downloaded clean, verified clean against the catalog sha256 — and then died
before generating a single token:

```text
error: malformed bpe-vocab model file: byte-level vocab has no single-byte
       token for byte 0x04; byte-level BPE requires all 256 bytes as base tokens
```

So the shipped default model could not be used at all. Every synthetic fixture in
the tokenizer suite passed throughout, because the fixtures were built the way
the code expects rather than the way HuggingFace actually ships.

The message named the symptom exactly and still left **two opposite** causes
open, needing opposite fixes:

1. **The loader is wrong.** llama.cpp writes `LLAMA_TOKEN_TYPE_BYTE` tokens as
   the literal text `<0x04>`. Read through the GPT-2 byte-level alphabet that
   becomes six ASCII characters, not byte `0x04` — the byte would be *in* the
   file and we simply fail to see it. Fix: teach the loader to decode them.
2. **The check is too strict.** A BPE trainer only seeds the full 256-byte
   initial alphabet when told to; a vocabulary trained without that genuinely
   lacks bytes its corpus never contained. Fix: tolerate holes.

Guessing here is how a "fix" makes the next failure harder to read, so we did not
guess. `kopitiam models inspect` on the real 386 MB file answered it:

```text
tokens:  49152      merges: 48900
token types:  NORMAL 49135   CONTROL 17
missing single-byte tokens: 21
  0x04 0x06 0x13 0x14 0x16 0x1d 0xc0 0xc1 0xf1 0xf2 0xf5 0xf6 0xf7 0xf8 0xf9
  0xfa 0xfb 0xfc 0xfd 0xfe 0xff
verdict: vocabulary is genuinely incomplete: 21 byte(s) absent and NO `<0xNN>`
         spellings anywhere, so the file really has no token for them.
```

Zero BYTE-type tokens, zero `<0xNN>` spellings. Explanation (2), decided by the
file rather than by argument. This is a judgment call the maintainer would
normally make — it relaxes a documented invariant on a public API — so: AID.

## Decision

`BpeTokenizer::byte_ids` becomes `[Option<u32>; 256]`. A vocabulary missing
single-byte tokens **builds successfully**; at encode time an unmapped byte is
**dropped**. New `BpeTokenizer::missing_byte_tokens()` reports exactly which
bytes are absent, so the lossiness is discoverable rather than hidden.

### What tolerating actually costs — the precise part

`Tokenizer::encode` takes `&str`, so its input is **always valid UTF-8**. Of the
21 missing bytes, **13 cannot occur in valid UTF-8 at all**: `0xc0`/`0xc1` are
the forbidden overlong lead bytes, and `0xf5..=0xff` sit above the largest legal
lead byte `0xf4`. Their absence is free.

The 8 reachable ones are `0x04 0x06 0x13 0x14 0x16 0x1d` (C0 controls EOT, ACK,
DC3, DC4, SYN, GS) and `0xf1`/`0xf2` (lead bytes for U+40000..=U+BFFFF — planes
4–11, which Unicode leaves entirely unassigned). In practice: **six rare control
characters**.

So `decode(encode(s)) == s` still holds for every `s` free of those 8 bytes, and
for an `s` containing one, that byte silently vanishes from the round trip.

## Alternatives considered

* **Keep refusing (status quo).** Trades a narrow, documented lossiness for "the
  default model cannot be used at all". A model whose only sin is never having
  seen an EOT character is not a malformed model.
* **Substitute `<UNK>`.** Byte-level BPE has no `<UNK>` to substitute *with*.
  Inventing an id puts a token into the stream the model was never trained on —
  worse than a dropped control character, and harder to notice.
* **Synthesise the missing tokens** at load (append 21 new ids). They would have
  no embedding rows in the weights, so the first use indexes out of bounds or
  reads garbage. Strictly worse than dropping.
* **Error only above some threshold** ("more than N bytes missing = malformed").
  No principled value of N exists. A guessed constant that rejects real models is
  the same bug again, just further away.
* **Fix the loader instead** (explanation 1). Ruled out **by evidence, not
  preference**: the file has no `<0xNN>` tokens to decode. Kept live in
  `models inspect` for other files, since some GGUFs really are that case.

## What would make this wrong

* **A GGUF that IS the loader-gap case gets quietly tolerated.** If a file has
  `<0xNN>` tokens we fail to decode, we now build a lossy tokenizer instead of
  erroring — silent quality loss instead of a loud failure. Mitigation:
  `models inspect` still distinguishes the two and says so. Stronger fix, if this
  ever bites: have the loader decode BYTE-type tokens, then holes are only ever
  genuine.
* **A hole in a byte that real text actually uses.** The argument above rests on
  the missing set being controls and unassigned-plane lead bytes. A vocab missing,
  say, `0x41` (`A`) would silently mangle ordinary English, and this change would
  let it. Nothing here checks for that. If that model appears, the answer is a
  loud warning keyed on *which* bytes are missing, not a return to all-or-nothing.
* **Dropping turns out to be the wrong reference behaviour.** We could not verify
  HuggingFace `tokenizers`' exact no-`unk_token` behaviour from source — it is
  not vendored here and this box's egress is limited. The choice is argued from
  first principles above, not copied. If upstream demonstrably does something
  else, revisit: a tokenizer that disagrees with the one a model was trained with
  is a correctness bug, not a style difference.
* **Somebody depends on the old error.** Nothing in-tree did (checked), but this
  is a public API relaxing a documented guarantee. It is a semver-visible
  behaviour change for any external caller relying on the strictness.

## Validation

* 4 new unit tests in `bpe.rs`: a holed vocab builds; a complete one reports no
  holes; text avoiding the holes round-trips byte-exact; a hole byte is dropped
  with its neighbours encoded identically and no bogus id invented.
* `kopitiam-tokenizer`: 59 tests green, clippy `-D warnings` clean.
* **The real proof** — `netfetch_end_to_end` against the actual downloaded
  weights, which is what a synthetic fixture could never do:

  ```text
  prompt "The capital of France is" -> " Paris."
  PASS  smollm2-360m-instruct-q8_0   reached generate
  ```

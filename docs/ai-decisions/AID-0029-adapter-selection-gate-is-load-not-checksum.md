# AID-0029: The CLI's local-model selection gate is "LocalAdapter::load succeeds", not "ModelStore::verify passes"

* **Status:** Pending review
* **Bead:** `kopitiam-oii`
* **Date:** 2026-07-16
* **Decided by:** AI (Claude), maintainer absent

## The decision

When `apps/cli` picks which `kopitiam_ai::ModelAdapter` to hand a workflow
(`plan`, and later friends), it selects the real on-CPU
`kopitiam_ai::LocalAdapter` whenever a `.gguf` is **present on disk and
`LocalAdapter::load` succeeds against it**. It deliberately does **not** gate on
`kopitiam_models::ModelStore::verify` (the SHA-256 checksum gate), and it
deliberately does **not** call `ensure_available` (which would autofetch).

## Why this was a judgment call

The task brief says "if a *verified* model file is on disk, build a
`LocalAdapter::load(path)`." Taken literally, "verified" points at
`ModelStore::verify`, the checksum gate. But the shipped catalog
(`kopitiam_models::Catalog::builtin`) carries **placeholder** sha256s (64
zeros), on purpose, because real hashes need a network download at authoring
time that has not happened yet. So `verify` against a *real* BYO `.gguf` dropped
at the store path returns `ChecksumMismatch` today — a real file fails the
checksum gate precisely because the catalog's recorded hash is a sentinel, not
because the bytes are bad.

Gating selection on `verify` would therefore mean: **no local model can ever be
selected until the maintainer records real hashes** (a network, maintainer-only
step). That defeats the whole point of the task — closing the loop so a BYO
`.gguf` actually runs — and it would do so silently.

## What was decided, and the alternatives

**Chosen:** presence-on-disk plus `LocalAdapter::load` succeeding is the gate.
`load` parses the GGUF, builds the `QwenModel`, and builds the tokenizer — it is
the real, honest answer to "can we actually run this file." A file that loads is
runnable; a file that does not degrades cleanly to Echo with the loader's error.

Alternatives considered:

* **Gate on `ModelStore::verify`.** Rejected: with placeholder checksums it
  selects *nothing*, ever, so end-to-end inference stays dead until an unrelated
  maintainer-only step happens. Honest provenance, useless selection.
* **Call `ensure_available` (autofetch) during `plan`.** Rejected twice over:
  (1) it would silently download hundreds of MB the first time someone runs a
  workflow command — a rude surprise for a "fast, offline" command; and (2) with
  placeholder checksums it fetches and then *fails* the gate anyway, so it buys a
  slow error, not a model. Autofetch stays an explicit user action
  (`kopitiam models pull`).
* **Skip verification, load anything present, no fallback message.** Rejected:
  the fallback must be explained, or an offline user is stranded with no idea how
  to get a model.

## What would make this wrong

* If real checksums get recorded in the catalog **and** provenance verification
  becomes a hard requirement before running *any* weights (e.g. a supply-chain
  policy: "never execute a model whose bytes we did not vouch for"), then the
  gate should tighten to `verify`-passes, with a BYO override for files the user
  vouches for themselves. Today, with placeholder hashes, that policy would just
  mean "never run anything," so it is not yet the right call.
* If `LocalAdapter::load` were ever cheap-but-incomplete (accepts files it
  cannot actually generate from), "load succeeds" would be too weak a gate. It
  is not — `load` builds the full model and tokenizer — but a future refactor
  that made `load` lazy would break this assumption.
* If autofetch-on-first-use were later judged the *desired* UX (download the
  default model the first time `plan` runs), the "never `ensure_available` here"
  half of this decision would need revisiting — but only once real checksums
  exist, or the download is guaranteed to fail the gate.

## Note on placeholder checksums

The real-hash recording is a separate, maintainer-driven, network-requiring step
that this session could not do (no network, and the brief forbids downloading).
It is filed as its own bead. Until it lands, end-to-end inference through the CLI
can only be exercised with a BYO real `.gguf` via `KOPITIAM_MODEL_GGUF` or
dropped at the store path.

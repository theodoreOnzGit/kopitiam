# ADR-0001 — Talk to `ollama serve` as a client; abandon the in-process port

**Status:** Accepted
**Date:** 2026-08-26
**Decided by:** the maintainer
**Supersedes:** `AID-0055-ollama-port-crate-boundary.md` (lived only on the
abandoned `ollama-port` branch, never reached `main`)
**Preserved at:** `archive/ollama-port` (tip `c0a819b`, 24 commits, ~82k insertions)

## The decision

KOPITIAM does **not** carry its own port of ollama. It talks to a **running
`ollama serve`** over ollama's REST API, as a client.

Maintainer, in their own words:

> "let's not try porting ollama ... rewriting the matrix tensors and gguf
> readers is a pain ... needs too much debugging. rather rely on ollama server
> and we do client side"

## What was actually tried first

The `ollama-port` branch got a long way: 24 commits, ~82k insertions, a whole
`kopitiam-ollama` crate — envconfig, the content-addressed blob store with real
sha256, GGUF `GraphSize` arithmetic, model parsers and renderers for a dozen
model families, the registry pull/push path, the scheduler, routes, the OpenAI
compat layer, and a safetensors→GGUF converter covering 10 architectures.

It was not abandoned for lack of progress. It was abandoned because the part
underneath all of it — **the matrix/tensor kernels and the GGUF readers** — is
where the debugging cost lives, and that cost is unbounded in a way the rest of
the port is not. A renderer that is subtly wrong produces a visibly wrong
prompt. A quant block layout that is subtly wrong produces *plausible garbage*,
and every model, quant format and architecture is a fresh chance to be subtly
wrong. `bd-p1d`-style flakiness is annoying; silently-wrong numerics are worse,
because they look like working software.

The evidence was already on the tree before this decision: the Q4_K_M bug
(Q8_0 fine, Q4_0 fine, **only** Q4_K_M broken) took a whole session to localise
and is exactly this genre of failure.

## Why a client is the right shape

* ollama already solved it, is maintained by people whose full-time job it is,
  and is tested against far more models and hardware than KOPITIAM will ever
  reach. This is `CLAUDE.md`'s "prefer an existing implementation over writing
  a new one" hard rule applied at the *process* boundary rather than the crate
  boundary.
* The API surface we need is small and stable — `/api/chat`, `/api/generate`,
  `/api/tags`, `/api/show` — versus ~82k lines of port to keep in step with
  upstream forever.
* It is still **local**. `ollama serve` runs on the same machine, offline. The
  "Offline First" pipeline (existing knowledge → native Rust → local AI → cloud
  AI) is intact; local AI just moves out-of-process.
* We already borrow ollama's judgment anyway: `kopitiam-ai/src/local/generation.rs`
  transcribes its `DefaultOptions()` sampling defaults verbatim, with provenance.
  Depending on the real thing is more honest than half-copying it.

## The cost, stated plainly

**This is a genuine departure from the Pure Rust Core promise, and it should
not be recorded as though it were free.**

`CLAUDE.md`'s promise is that `git clone` + `cargo build` just works, with
nothing extra to install. A user who wants local inference must now install
ollama — a Go binary, not a crate, not reachable through cargo. That is a real
install-step, on a platform matrix (Android/Termux especially) where it may not
be a comfortable one.

What keeps this acceptable rather than a violation:

* It is an **optional accelerator**, not a build dependency. The workspace still
  compiles and every non-inference feature still runs with ollama absent. The
  core stays pure Rust; only local inference gains an external process.
* It is **local and offline**, so the local-first principle — the one the Pure
  Rust Core rule exists to protect — survives intact.

What would make this decision wrong:

* If ollama's API turns out to be unstable across releases, we have traded a
  bounded debugging cost for an unbounded chasing-upstream cost, and we have
  gained nothing.
* If Android/Termux cannot run `ollama serve` acceptably, then on the platform
  KOPITIAM most cares about being portable to, local inference is simply gone —
  and the in-process path, painful as it was, was the only one that could have
  worked there.
* If KOPITIAM ever needs inference behaviour ollama does not expose (custom
  sampling, direct KV-cache access, embedding internals), the client boundary
  becomes a ceiling rather than a convenience.

Any of those three turning true is grounds for a follow-up ADR, and
`archive/ollama-port` is deliberately kept so that follow-up has somewhere to
start from rather than starting cold.

## Consequences

* `crates/kopitiam-ai/` gains an ollama-client adapter beside `cloud.rs`.
  `src/local/` — the in-process path — is untouched by this decision and keeps
  working for whatever it already handles.
* `archive/ollama-port` holds the work. The `ollama-port` branch itself is
  still present — the session that made this decision lacked permission to
  delete refs (HTTP 403), so it is left for the maintainer:
  `git push origin --delete ollama-port`. Deleting it loses nothing; the
  archive ref is the same commit.
* `gh-73` (the AID-0055 number collision between the branch and `main`) is moot
  and closed: the branch's AID never reaches `main`, so `main`'s
  `AID-0055-type1-charstrings-from-spec-not-hayro.md` keeps the number.
* `crates/kopitiam-runtime/vendor/ollama` stays where it is — still valuable as
  read-only reference (it is where our sampling defaults are cited from), and
  inert per `CLAUDE.md`'s vendored-code rule.

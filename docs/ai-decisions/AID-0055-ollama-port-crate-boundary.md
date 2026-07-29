# AID-0055: The ollama port lands in its OWN crate, below `kopitiam-ai`, and reimplements Go's `text/template` rather than transliterating it

* **Status:** Pending review
* **Date:** 2026-07-29
* **Decided by:** AI (Claude), maintainer present and directing the *what*, not the *where*
* **Bead:** `bd-uc9`
* **Scope:** the new crate `crates/kopitiam-ollama/`, its dependency position in
  the workspace, and the decision to write a Go-template engine instead of
  reusing / transliterating one. Does **not** change the AGPL-3.0-only licence,
  the Pure Rust Core rule, or the attribution rules in
  `docs/ACKNOWLEDGEMENTS.md` — it complies with all three.

## The brief

The maintainer asked, in three messages: *"port ollama to rust in kopitiam"*,
*"kopitiam ai"*, and — the standing rule for the whole exercise —

> *"if in doubt, whether kopitiam-ai or ollama is correct, pick ollama. It is the
> golden oracle"*

plus *"I think it's best not to figure everything out when the source is already
there"*, i.e. **read the Go, do not reason it out from first principles.**

So the *what* was decided by the maintainer and is not in question here. What
was **not** specified, and what this AID records, is:

1. **Where** the port lives. `kopitiam-ai` already exists and is the obvious
   home by name.
2. **How** to handle the fact that ollama's most valuable piece — prompt
   templating — is built on Go's `text/template`, which has no Rust equivalent
   and is not itself part of ollama.

## The decision

**1. A new crate, `kopitiam-ollama`, positioned BELOW `kopitiam-ai`, depending on
nothing else in KOPITIAM.**

Everything ported is deterministic text/byte work: parse a name, parse a
Modelfile, render a template, resolve a blob path, hold the sampling defaults.
None of it needs a model, a GPU, or a socket. Keeping it in a crate with no
KOPITIAM dependencies means **the layer that decides what prompt the model sees
is testable without running a model** — 127 tests, no fixtures, no weights,
sub-second. That property is worth a crate boundary on its own.

Three further reasons:

* **Provenance is legible.** One crate, one upstream, one licence relationship.
  `docs/ACKNOWLEDGEMENTS.md` can say "this crate is a port of ollama" in one
  sentence instead of tracing which parts of a mixed crate are derivative.
* **`kopitiam-ai` keeps its role.** `kopitiam-ai` owns the `ModelAdapter` seam —
  the one door a model is reached through, cloud or local. Folding a Go
  runtime's serving layer into it would blur that seam into a grab bag.
* **The architecture rule in CLAUDE.md survives.** Nothing below
  `kopitiam-workflow` may depend on `kopitiam-ai`. A port crate that depended on
  `kopitiam-ai` would either violate that or force `kopitiam-ai` to depend on
  the port and inherit its whole surface. Below-and-independent avoids both.

**2. `gotmpl` reimplements Go's `text/template` semantics; it does not
transliterate Go's engine.**

Go's implementation is ~7000 lines, and most of that mass exists to bridge a
static type system into a dynamic template language via reflection. KOPITIAM's
template data is a small closed enum, so that machinery has nothing to do here.
What is ported *exactly* is the **language semantics**, because those are where
"what Rust would naturally do" is wrong:

| Semantic | Why a Rust-native instinct breaks it |
| --- | --- |
| Go truth (`""`, `0`, empty list are false) | `{{ if .System }}` means "if there IS a system prompt"; an `Option` check renders the wrong branch |
| Maps range in **sorted key order** | Go sorts deliberately for reproducibility; a `HashMap` gives a different prompt every run |
| `and`/`or` return a **value**, not a bool | `{{ or .A .B }}` prints A |
| `else if` **nests** | A chain closes with one `{{ end }}`; treating it as flat fails to parse every real chat template |
| `missingkey=zero` | ollama sets this option; a missing field must be nil, not an error |
| Trim markers `{{- ` / ` -}}` | A stray newline before `<|im_start|>` degrades output without ever looking like a bug |

Each is attributed and tested at the point of implementation.

**Scope was cut deliberately and loudly**: `{{ template }}`, `{{ define }}`,
`{{ block }}` are **rejected with an error**, not silently mis-rendered. No chat
template uses multi-template composition; if one ever does, that is a bug to fix
in `gotmpl`, not a reason to fall back to hardcoded ChatML.

## Alternatives considered

* **Fold it all into `kopitiam-ai`.** Rejected: mixes derivative and clean-room
  code in one crate (bad for provenance), grows `kopitiam-ai` past its
  `ModelAdapter` remit, and drags the offline-testable template layer behind a
  crate that pulls in cloud adapters and an inference stack.
* **Split across existing crates** — names + store into `kopitiam-models`,
  templates into `kopitiam-ai`, options into `kopitiam-runtime`. Rejected: it
  scatters one upstream across four attribution sites, and any future ollama
  change has to be re-diffed against four places.
* **Use an existing Rust template crate** (`tera`, `handlebars`, `minijinja`).
  Rejected on the oracle rule. These are Jinja/Handlebars-shaped, not Go-shaped;
  the differences are exactly in truth semantics, map ordering, and `else if`
  handling — i.e. precisely the things that must match. A template that renders
  *almost* right produces a prompt that is *almost* what the model was trained
  on, and that failure is silent.
* **Transliterate Go's `text/template` wholesale.** Rejected as cost with no
  benefit: the reflection layer has no counterpart in a closed-enum data model,
  and porting it would mean maintaining thousands of lines that can never be
  exercised.
* **Wait and vendor ollama's `.gotmpl` library files too** (upstream's `Named()`
  picks a bundled template by Levenshtein distance across ~20 embedded files).
  Deferred, not rejected: KOPITIAM reads the template out of the GGUF, which
  covers the real case. Tracked on `bd-uc9`.

## Why this is the maintainer's call, and why it went this way

Crate boundaries are close to irreversible once other crates depend on them, and
this one also fixes where a derivative work sits in a copyleft tree — both
maintainer-grade decisions. The maintainer was present for the *goal* but did not
specify the shape, and stalling to ask would have blocked the whole port on a
question that has a defensible default.

The default chosen is the conservative one: **a new leaf crate changes nothing
that already exists.** No current crate gained a dependency, no existing API
moved, and `cargo check --workspace` is unchanged. If this AID is reversed, the
port can be dissolved into other crates later at the cost of a mechanical move —
whereas having polluted `kopitiam-ai` first would not be undoable so cheaply.

## What would make this wrong

* **If `gotmpl` turns out to need most of Go's engine anyway.** The bet is that
  chat templates use a small, stable corner of the language. If real GGUFs start
  needing `{{ template }}`, method calls, or the wider `fmt` verb set, the
  "subset" framing was wrong and either a fuller port or a different strategy is
  needed. *Signal:* `gotmpl` parse errors on templates from real models.
* **If a maintained Rust crate appears that reproduces Go template semantics
  faithfully** — including sorted map iteration and Go truth. Then this is
  several hundred lines of avoidable maintenance. *Signal:* such a crate exists
  and passes the semantics table above.
* **If the crate stops being dependency-free.** The offline-testability argument
  is the load-bearing one. The moment `kopitiam-ollama` needs `kopitiam-runtime`
  or `kopitiam-gpu` — plausibly at the scheduler / memory-fit stage of the port —
  the boundary should be re-argued rather than quietly relaxed, because a
  scheduler is a *different* kind of thing from a text layer and may deserve its
  own crate again.
* **If the maintainer wanted the port to REPLACE `kopitiam-ai`'s local path
  rather than sit beside it.** Nothing is wired into `kopitiam-ai` yet; the port
  is additive so far. If the intent was a swap rather than a new foundation, the
  integration step differs and should be settled before it is built.
* **If "ollama is the oracle" was meant narrowly** — only for behaviour
  KOPITIAM already implements — rather than as a licence to port whole
  subsystems. This AID reads it broadly. The maintainer can narrow it.

# AID-0039: HuggingFace fetching is a default-on capability, riding the existing `net` gate

* **Status:** Pending review
* **Bead:** `kopitiam-9l4`
* **Date:** 2026-07-18
* **Decided by:** AI (Claude), maintainer absent

## The decision

`kopitiam-models` now ships a first-class HuggingFace fetcher (`hf::HfFetcher`)
plus an HF declaration type (`hf::HfModel` / `HfFile` / `Revision`). HF fetching
is **on by default**: it rides the crate's already-existing `net` feature, which
is already in `[features] default = ["net"]`. So a normal `cargo build` /
`cargo install` gets HF fetching out of the box -- practically non-optional for
users, which is what the maintainer asked for ("non-optional... a core,
always-available capability").

The offline escape hatch stays exactly as it was: `--no-default-features` still
gives a pure bring-your-own, zero-network, zero-TLS, zero-`ring` build that
compiles and passes the offline subset of the test suite. HF fetching did **not**
get its own separate feature flag, and the `net` gate was **not** removed.

## Why this was a judgment call

The maintainer's word was "non-optional", and the existing crate has a `net`
feature whose whole reason to exist is that the catalog / verification core can
compile and test **offline** (CI, the Pure-Rust-minimal build). Those two pulls
are in tension: "always available" vs "must still build with zero network deps".
Reconciling them is a design call the maintainer would normally make, and they
were not here to make it.

**Challenge-the-premise note.** The task literally instructed "put `net` in
`[features] default = [...]`". That was **already true** in the crate before this
work (a prior sprint set `default = ["net"]`). So no `Cargo.toml` feature change
was needed -- the literal instruction was already satisfied. The real decision
left to make was *how* HF fetching relates to that gate, not whether to flip it.

## What was decided, and the alternatives

**Chosen:** HF fetching is default-on **by riding the existing `net` feature**.
No new `hf` feature. `HfFetcher` is `#[cfg(feature = "net")]`, same as the
generic `HttpFetcher`. The pure, offline-safe half of the HF path -- URL
construction (`hf_resolve_url`), revision pinning (`Revision`), the
`HfModel -> ModelSpec` fold, token normalisation -- compiles with **no** feature
at all, so `--no-default-features` still gets the declaration types and can drive
the acquire path with a caller-supplied `Fetcher`.

Alternatives considered:

* **Keep HF behind its own opt-in feature (e.g. `hf`).** Rejected -- the
  maintainer explicitly wants HF core, not hidden behind an obscure flag. A
  separate flag is exactly the "obscure flag" they warned against.
* **Remove the `net` gate entirely, make the HTTP stack unconditional.**
  Rejected -- that kills the offline CI build and the Pure-Rust-minimal
  (`--no-default-features`) build, which is the whole point of the gate. It would
  also force `ureq` + `rustls` + `ring` (C + perlasm) into *every* downstream,
  including ones that want a genuinely dependency-free build. "Non-optional for
  users" does not have to mean "unconditional in the build graph".
* **A second HTTP implementation just for HF.** Rejected -- `HfFetcher` is
  `HttpFetcher` plus an optional `HF_TOKEN` bearer header and a pinned
  same-host redirect-auth policy. Reusing the same `ureq` stack and the same
  `Fetcher` seam keeps one network touch-point, not two.

## What would make this wrong

* **A downstream that MUST build with zero network deps and cannot pass
  `--no-default-features`.** If some consumer pulls `kopitiam-models` as a
  transitive dependency with `default-features = true` forced on (e.g. another
  workspace crate depends on it plainly and re-exports it), that consumer drags
  in `ureq`/`rustls`/`ring` whether it wants them or not. The mitigation exists
  today (depend with `default-features = false`), but if a hard "no C in the
  default build graph, full stop" policy ever lands, default-on `net` becomes the
  wrong default and the gate would need to flip back to off-by-default with HF
  fetching as an explicit opt-in.
* If the offline core ever stops compiling under `--no-default-features` (someone
  puts network-only code outside a `#[cfg(feature = "net")]`), the reconciliation
  silently breaks -- the offline test job is the guard against that and must stay.

## Notes

* `net` was already `default`; this AID records the *reasoning* for keeping HF
  fetching tied to that default rather than any change to the feature list.
* The catalog is **not** populated with specific HF models here -- the maintainer
  names those later. Only the mechanism (`HfModel`/`Revision` + `HfFetcher`)
  landed, with a `// maintainer will populate` marker in `Catalog::builtin` and a
  follow-up bead (`kopitiam-56q`).

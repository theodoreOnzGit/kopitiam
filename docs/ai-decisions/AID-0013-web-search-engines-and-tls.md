# AID-0013: which search engines `kopitiam-web` talks to, and the uncomfortable truth about "rustls"

* **Status:** Pending review
* **Bead:** `kopitiam-b4u`
* **Date:** 2026-07-14
* **Decided by:** AI (Claude), maintainer absent

Two judgment calls, both the maintainer's to make, both made in their absence
because stalling the crate on either would have blocked the whole thing.

1. **Which search APIs to implement**, of maybe eight candidates.
2. **What to do about the fact that `rustls` is not actually pure Rust**, which I
   did not expect and which the brief did not anticipate.

The second one is the important one. If you only read part of this, read part 2.

---

## Part 1: the engines

### What was decided

Implement **two** adapters, both behind non-default cargo features:

| Feature | Provider | Key? | Cost | Self-hostable |
| --- | --- | --- | --- | --- |
| `searxng` | **SearXNG** — *recommended* | no | free | **yes** |
| `brave` | Brave Search API | yes | free tier, then metered | no |

Default build: **neither**. No HTTP client, no TLS, no socket. `NullProvider`
(honest error) and `StaticProvider` (deterministic fixtures) are the only
providers that exist in a default build, and they need nothing.

### The verdict on SearXNG vs. renting a vendor

The brief asked me to rate this honestly and said SearXNG "deserves serious
consideration precisely because it can be self-hosted". I went further than
that: **SearXNG is the recommended provider, and Brave is the concession.**

The argument is not really about search quality. It is that KOPITIAM's founding
principle is that *no external service may ever be required*, and there is
exactly one way to satisfy that for web search: the user owns the engine.

* **No API key.** Nothing to obtain, leak, rotate, or bill.
* **No vendor.** No terms of service that can change under you; no account that
  can be closed; no company that can be acquired and shut down. CLAUDE.md says to
  assume a ten-year horizon. Most search-API companies will not see it.
* **No cost**, so the bottom rung of the Offline First ladder stops being a
  financial cliff.
* **AGPL-3.0** — the same licence as KOPITIAM. Not decisive, but not nothing:
  it is a project with the same theory of ownership.
* You can point it at a **private, institutional, or air-gapped** index.

The honest costs, which are real:

* **You have to run it.** A scientist who wants a search box, not a sysadmin's
  afternoon, will not do this. That is precisely why Brave is implemented too.
* **JSON must be switched on** in `settings.yml` (`search: formats: [html,
  json]`); a stock instance answers a JSON request with 403. The adapter says so
  in as many words rather than leaving the operator to guess.
* **Public instances are not a substitute.** Most disable JSON, rate-limit hard,
  and block bots — and pointing KOPITIAM at a stranger's server reintroduces
  exactly the third-party dependency self-hosting removed. Run your own.
* **It is a scraper underneath.** Upstream engines break it periodically. The
  failure is at least visible and honest.

### Why Brave, of the vendors

For the user who will not run infrastructure. Of the commercial options it is
the least bad fit:

* It has an **independent crawl and index**. This matters more than it sounds.
  Serper and SerpAPI *resell Google*; paying them buys a costlier copy of the
  same answer, not a second opinion. "Brave said X on 2026-07-14" is a claim
  about a distinct index, which is worth something in a provenance record.
* A **plain, documented, stable JSON endpoint** — not a scraping layer that
  breaks when someone edits a CSS class.
* A free tier, so it can be tried without a commitment.

### What was rejected, and why

**Tavily** — rejected, and this one is worth spelling out because it is
superficially the *most* attractive option: it is purpose-built for LLM/RAG use,
which is nominally our use case.

It is rejected **because** of that. Tavily's value-add is that it returns
LLM-processed, re-ranked, summarized answers. That is precisely the part
KOPITIAM must not buy. A provenance record has to say *what the source said*,
not what a vendor's model said the source said. Paying for a paraphrase and
then recording it as evidence, with a URL and a content hash attached, would
manufacture exactly the kind of laundered claim this crate is built to prevent.
The convenience is the defect.

**Serper / SerpAPI** — rejected. Google scraping proxies. Grey-area terms of
service (Google's ToS prohibit what they do; the legal risk is theirs until it
is yours), pure rent, and no independent index. Nothing to recommend them except
Google's result quality, which is not worth the dependency.

**DuckDuckGo** — rejected, on a factual point that trips people up constantly:
**DuckDuckGo has no web-search API.** `api.duckduckgo.com` is the *Instant
Answer* API, which returns topic abstracts and is empty for most real queries.
The thing people actually mean — scraping `html.duckduckgo.com` — violates their
ToS and breaks continually. Implementing it would have produced an adapter that
returns nothing most of the time, which in a crate whose central thesis is *"no
results" must never be confused with "could not search"* is close to the worst
possible outcome.

### What would make Part 1 wrong

* If self-hosting SearXNG proves too painful in practice for the actual users,
  and everybody ends up on Brave, then the philosophical win is theoretical and
  we should say so plainly rather than pretending the recommended path is the
  travelled one.
* If Brave's free tier disappears or its index quality collapses for scientific
  queries, the vendor adapter is dead weight and something else is needed.
* If a workflow genuinely needs full page *content* rather than snippets, none
  of this is sufficient and the fetch-and-extract problem has to be faced (see
  the bead; it is deliberately out of scope here).

---

## Part 2: "use rustls, never OpenSSL" — the instruction is right, the premise is incomplete

The brief said, with emphasis:

> **HTTP client: it MUST be rustls-backed, not OpenSSL.** [...] **Verify the
> feature flags actually exclude OpenSSL** — do not assume.

I verified. The instruction is satisfied. But verifying turned up something the
maintainer should know, because it is a real (if bounded) crack in the Pure Rust
Core:

> **`rustls` is not pure Rust.** It is a pure-Rust TLS *protocol* implementation,
> but it delegates all cryptography to a pluggable *provider*, and **every
> production-ready provider today contains C or assembly.**

Concretely:

| Provider | What it actually is | Used by |
| --- | --- | --- |
| `aws-lc-rs` | AWS-LC. **C.** | rustls' own default |
| `ring` | BoringSSL-derived. **C + perlasm.** | `ureq`'s `rustls` feature |
| `rustls-rustcrypto` | genuinely pure Rust | nobody; its own README says it is not production-ready |

So "rustls, not OpenSSL" does not get you to zero C. It gets you off *OpenSSL*,
which is a different and still very worthwhile goal — and, notably, the goal the
maintainer actually named, for the reason they named (Android).

### What was decided

Use **`ureq` 3 with rustls (hence `ring`)**, and be honest about it in the
`Cargo.toml`, in the module docs, and here.

```toml
ureq = { version = "3", default-features = false, features = ["rustls"], optional = true }
```

Verified, not assumed:

* `ureq`'s `rustls` feature expands to `rustls-no-provider` + `_ring` +
  `rustls-webpki-roots`. The provider is `ring`.
* `ureq`'s `native-tls` feature — the one that pulls `openssl-sys` — is **not
  enabled**, which is what `default-features = false` guarantees.
* `cargo tree -p kopitiam-web --features brave,searxng | grep -iE
  'openssl|native-tls'` prints **nothing**.

### Why `ring` is acceptable here, specifically

1. **It is behind an off-by-default feature.** The core platform compiles zero
   bytes of it. `cargo build -p kopitiam-web --no-default-features` and the
   default build are identical, and both contain no network stack at all. The
   Pure Rust Core rule says *"the core platform should remain entirely buildable
   using Cargo"* and *"optional integrations are acceptable"* — this is the
   textbook case of an optional integration.
2. **It solves the problem the maintainer actually has.** They named Android and
   OpenSSL together, and they were right to: OpenSSL's build system is the thing
   that reliably breaks Android cross-compilation. `ring` cross-compiles to
   `aarch64-linux-android` cleanly. Swapping a C library that breaks the target
   for a C library that does not is a real win even though it is not the
   *stated* win.
3. **The alternative is worse.** Shipping `rustls-rustcrypto` would mean putting
   an explicitly-not-production-ready crypto implementation on the network path
   in order to satisfy a rule by its letter. That is a security decision dressed
   up as a purity decision.

### What would make Part 2 wrong

Any of these, and the decision should be revisited:

* **A pure-Rust rustls provider becomes production-ready.** `rustls-rustcrypto`
  maturing, or something like it. Then there is no trade-off left to make and we
  should take it. This is the most likely way this AID ages badly, and it would
  be a happy ending.
* **The maintainer's rule is stricter than I read it** — i.e. "no C anywhere in
  the tree, ever, including in optional features". If that is the actual bar,
  then *no* usable TLS stack exists in Rust today, and the correct conclusion is
  that `kopitiam-web` should have **no live adapters at all** — just the trait,
  `NullProvider`, `StaticProvider`, and the cache. That would still be a
  coherent and defensible crate (arguably a purer one), and it is a small
  deletion away. Say the word.
* **A target appears where `ring` does not build.** Some exotic embedded or
  wasm-adjacent target. Then the feature simply is not enabled there, which is
  the point of it being a feature.

### One more thing worth flagging

The two dependencies I added outside the network feature — `chrono` (provenance
timestamps) and `sha2` (content hashes) — are both **already in this workspace's
`Cargo.lock`** (via `kopitiam-mux`, `lopdf`, `rmux`), so they add no new
supply-chain surface. `chrono` is pulled with `default-features = false` to
exclude `iana-time-zone` (which reaches for Objective-C on Apple and JNI on
Android); provenance is recorded in UTC only, so local time is not merely
unnecessary but actively unwanted — a retrieval timestamp must mean the same
thing to every reader of the knowledge graph.

Neither is declared in the workspace `[workspace.dependencies]` table, because
that file belongs to the maintainer and fourteen agents were sharing the repo.
They are declared directly in `crates/kopitiam-web/Cargo.toml`. Promoting them
to workspace dependencies would be tidier and is a two-line change.

# AID-0005: how kvim gets a language server on Android

* **Status:** Pending review (rust-analyzer decision confirmed by maintainer)
* **Bead:** `kopitiam-cj0.9`
* **Date:** 2026-07-14
* **Decided by:** maintainer (rust-analyzer) + AI (the other two, explicitly delegated)

## Background

AID-0003 assumed kvim could fetch prebuilt static LSP binaries per platform,
sidestepping Mason's npm/pip/go shell-outs. Implementation then verified against
the GitHub releases API that **none** of the three servers the maintainer uses
publishes an `aarch64-linux-android` build. The download strategy has nothing to
download on the target platform it exists to serve.

The maintainer then supplied the key missing fact:

> I believe android has its own rust analyzer binary, I was able to install
> cargo in android. Just use that rust binary. I leave the other two decisions
> for the lsp to you.

That reframes the problem. **If cargo runs on Android, then any language server
that is itself a Rust program can simply be `cargo install`ed there.** Two of
the three are.

| Server | Implemented in | Obtainable on Android? |
| --- | --- | --- |
| rust-analyzer | **Rust** | Yes — Termux packages it, and `rustup component add` / `cargo install` both work |
| texlab | **Rust** | Yes — `cargo install texlab` |
| lua-language-server | C++ + Lua (LuaJIT) | Not straightforwardly — no Bionic release, needs a real C++ toolchain |

## What was decided

**A four-tier acquisition ladder**, tried in order, replacing the previous
binary download-or-give-up:

1. **`OnPath`** — already installed; use it. Unchanged, and still first: this is
   what makes the maintainer's *desktop* (where rust-analyzer is already on
   PATH) and their *Termux* (`pkg install rust-analyzer`) both just work with no
   network access at all.
2. **`SystemPackage`** — the platform's own package manager, where one exists
   and actually carries the server. On Android that means Termux's `pkg`, which
   is **not** GitHub Releases and is exactly the source AID-0003 failed to
   consider. This is *not* a reintroduction of the Mason failure mode: Mason's
   sin was shelling out to *language* toolchains (npm/pip/go) that are absent on
   the platform; `pkg` is the platform's native package manager and is present
   by definition.
3. **`CargoInstall`** — for servers that are Rust programs. Legitimate precisely
   because the maintainer has cargo on the device. It is slow (it compiles), so
   it ranks below a prebuilt package, and kvim says so before starting rather
   than appearing to hang for twenty minutes.
4. **`Download`** — prebuilt static binary, for the desktop targets that publish
   one. Demoted from first to last resort in the ladder's logic (it is still the
   fastest path on desktop; it is simply the one that cannot serve Android).

Anything with no tier available still reports **`Unavailable` with an honest
reason**, never a fabricated URL.

**Per-server outcomes on Android:**

* **rust-analyzer** — PATH, then Termux `pkg install rust-analyzer`, then
  `rustup component add rust-analyzer` / `cargo install`. This is the
  maintainer's confirmed decision.
* **texlab** — PATH, then Termux pkg, then `cargo install texlab`. My call, and
  an easy one: it is a Rust program, so the same reasoning that solves
  rust-analyzer solves it.
* **lua-language-server** — PATH, then Termux pkg (Termux does carry it), then
  **honestly unavailable**. My call. I am not going to pretend a C++/LuaJIT
  build bootstraps cleanly on a phone. Two mitigations worth noting rather than
  building today:
  * Lua is the *least* important of the three here — the maintainer's Lua editing
    is largely their Neovim config, and kvim's whole point is that it needs no
    Lua config.
  * KOPITIAM is already committed to a pure-Rust Lua 5.1 VM (`kopitiam-lua`,
    kvim Phase 4). A Rust-native Lua language server built on that would solve
    this properly, and would be ours. Filed as a follow-up, not scoped now.

## What would make this wrong

* If Termux does **not** in fact package rust-analyzer or lua-language-server
  under those names, tier 2 is dead weight for them (harmless — it falls through
  to tier 3 or to an honest `Unavailable`, it does not break anything).
* If `cargo install rust-analyzer` on a phone is unbearably slow in practice
  (it is a large crate), tier 3 is technically correct but practically useless,
  and tier 2 becomes load-bearing. The maintainer will find out faster than I can
  reason about it.
* I have **not** verified Termux's package names against Termux's actual repo —
  no Android device here to check against, and I declined to guess at a URL a
  second time. The names are marked in the code as unverified, and the failure
  mode if they are wrong is a clean fall-through, not a crash.

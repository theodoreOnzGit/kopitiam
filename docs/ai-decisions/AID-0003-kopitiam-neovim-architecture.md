# AID-0003: kopitiam-neovim — Lua compatibility, Android, and what `kvim` actually is

* **Status:** **Decision 1 CONFIRMED by the maintainer (2026-07-14): build the pure-Rust Lua VM.** `mlua` is rejected. The rest of the AID remains pending review.
* **Bead:** `kopitiam-nvim` (epic)
* **Date:** 2026-07-14
* **Decided by:** AI (Claude), maintainer absent

This is the most consequential AID so far. Read the first section even if you
skip the rest — I believe the brief contains a false premise, and I have
planned around what I think you actually want rather than what you literally
said. If I'm wrong, this is the one to reverse.

## The brief

> Make a `kopitiam-neovim` that rewrites the entire Neovim in Rust. It should
> be compatible with Lua addons. The CLI app should come preset with my Neovim
> behaviour by default in `~/.config`. HOWEVER, I want all plugins and even
> Mason to be rewritten in pure Rust **because it doesn't work on Android**.
> All Rust-based plugins should be written into the `kopitiam-neovim` crate.
> Which, when published, should be runnable with `kvim`.

## The false premise, stated plainly

**Mason does not fail on Android because it is written in Lua. It fails
because it shells out to `npm`, `pip`, `go install`, and `cargo` to fetch
language servers, and those toolchains aren't there.** Lua itself is ~20k
lines of C89 and cross-compiles to the Android NDK without complaint — it is
one of the most portable pieces of software ever written, and is embedded in
countless Android apps.

This matters because the two goals in the brief pull in opposite directions:

* "Compatible with Lua addons" requires **running Lua** — a real VM, with real
  metatables, coroutines, and `string.format`. Your 20 plugins are ~200k lines
  of Lua across telescope, neo-tree, blink.cmp, LuaSnip, plenary, and friends.
* "Rewrite all plugins in pure Rust" means **not running their Lua at all.**

If every plugin is rewritten in Rust, Lua compatibility is only needed for
*your config* and for third-party plugins you haven't rewritten yet. And if
Lua compatibility is a hard requirement, then a Lua VM must ship — at which
point the question is only whether that VM is C or Rust.

So: **the Android problem is a supply-chain problem, not a language problem.**
Solving it by rewriting plugins in Rust is right. Solving it by avoiding Lua
is solving the wrong thing.

## What was decided

**1. Ship a Lua VM. Target a pure-Rust one, but do not block on it.**

There is, today, no production-grade pure-Rust Lua interpreter.
`piccolo` (ex-`luster`) is a genuinely promising stackless VM but is
explicitly incomplete and has no full stdlib; `hematita` is unmaintained;
`full_moon` is a parser, not a VM. `mlua`/`rlua` are excellent and are
C bindings — which the Pure Rust Core rule in CLAUDE.md rejects for the
*core platform*.

The `crates/kopitiam-lua` stub already exists in this workspace, which I read
as prior intent to own this layer. Decision:

* `kopitiam-lua` becomes a **pure-Rust Lua 5.1 interpreter** (5.1 because that
  is the dialect Neovim's LuaJIT targets, and what every plugin is written
  against). It is a large but *bounded, well-specified* problem — the Lua 5.1
  reference manual is 100 pages and the language is famously small.
* Until it is complete, `kopitiam-neovim` is developed against it behind a
  trait (`LuaEngine`), with an **optional, non-default** `mlua` feature flag
  for developers who want to run real plugins today. The default build stays
  pure Rust and Cargo-only. The core platform's promise is kept; the escape
  hatch exists but you have to ask for it.

This is the decision most likely to be wrong, and the most expensive if it is.
Writing a Lua VM is a serious multi-month project on its own. **If you would
rather just take the `mlua` dependency and get a working editor a year
sooner, say so and I will reverse this** — it is a defensible call, and CLAUDE.md
does permit "optional integrations."

**2. Replace Mason with a pure-Rust LSP acquisition layer — this is the real
Android fix.** A new `kopitiam-lsp` (the stub also already exists) gains:
   * a registry of language servers with **direct download URLs for
     prebuilt static binaries** (no npm/pip/go), fetched over pure-Rust TLS
     (`rustls`, not OpenSSL — OpenSSL is the *other* thing that breaks on
     Android);
   * for Rust-implemented servers, the option to **statically link or vendor
     them** so nothing is downloaded at all;
   * per-platform/per-arch resolution including `aarch64-linux-android`.
   This is what makes `kvim` work on your phone. It is independent of the Lua
   question and should be built first.

**3. Rewrite the plugins in Rust, natively — not as "Lua plugins written in
Rust."** Your 20 plugins collapse into far fewer native subsystems, because
much of what they do is glue that a native editor simply *has*:

   | Your plugin | Becomes |
   | --- | --- |
   | telescope + plenary | native fuzzy finder (`nucleo` — pure Rust, by the Helix authors) |
   | neo-tree + nui + web-devicons | native file tree |
   | blink.cmp + LuaSnip | native completion + snippet engine |
   | mason + mason-lspconfig + nvim-lspconfig | `kopitiam-lsp` (decision 2) |
   | harpoon | native mark/jump list |
   | hop.nvim | native label-motion |
   | vim-airline + themes | native statusline |
   | vim-fugitive | native git integration (`gix` — pure Rust) |
   | aerial.nvim | native symbol outline (from LSP, which we already speak) |
   | gruvbox | native theme (colorschemes are data, not code) |
   | vim-easy-align | native align command |
   | lazy.nvim | **deleted** — nothing to lazily install if plugins are built in |
   | claudecode.nvim | native Claude integration (KOPITIAM already has `kopitiam-ai`) |

   Note `lazy.nvim` disappearing: a plugin *manager* is only necessary because
   plugins are third-party downloads. Built-in plugins need configuration, not
   installation. This is a real simplification, not a hand-wave.

**4. ~~`kvim` is a binary in `apps/`~~ — OVERRIDDEN BY THE MAINTAINER.**

   *Original decision (superseded):* put the binary in `apps/kvim` per
   AID-0001's apps-are-clients rule, with `kopitiam-neovim` as a pure engine.

   *Maintainer's instruction, received mid-session:* "when i do
   `cargo install kopitiam-neovim`, and run `kvim`, it should have the same
   behaviour as my neovim." That is explicit: the binary ships **from the
   `kopitiam-neovim` crate itself**, via a `[[bin]] name = "kvim"` target.
   Done that way. This is a deliberate, maintainer-authorized exception to
   the apps/crates split — noted here so nobody "fixes" it later.

   Consequence: `kopitiam-neovim` will get large. When a subsystem earns its
   own crate (the Lua VM already has one; the text engine probably will), it
   gets promoted out. I will not pre-split it into fifteen empty crates.

**5. Your config is vendored as the default, not read from `~/.config`.** You
asked for `kvim` to "come preset with my Neovim behaviour by default in
`~/.config`." Ambiguous, so: `kvim` ships your keymaps, settings, and plugin
choices **compiled in as the default configuration**, and *also* reads
`~/.config/kvim/` to override them. It will not write to or depend on
`~/.config/nvim/` — silently colonizing the real Neovim's config directory
would be hostile, and would break your actual Neovim. If you meant "read my
existing `~/.config/nvim` and just work," that's a config *importer*, which I
filed as a separate bead rather than assuming.

## Scope honesty

Neovim is ~1.3 million lines (C + Lua + Vimscript) with 30 years of accreted
behaviour. "Rewrite the entire Neovim" is not a task, it is a program of work
measured in years — Helix and Zed and Lapce are each many engineer-years and
none of them is Vim-compatible. I have phased it (see beads) so that each
phase is independently useful:

* **Phase 1** — text engine: rope buffer, undo tree, marks, multi-cursor-ready.
* **Phase 2** — modal editing: modes, motions, operators, registers, macros,
  text objects, ex commands. *At this point it is a usable vi.*
* **Phase 3** — `kopitiam-lsp` + Android-capable server acquisition. *Now it's
  a usable IDE, and it runs on your phone.*
* **Phase 4** — `kopitiam-lua` VM + the `vim.*` API surface. *Now Lua configs
  and third-party plugins load.*
* **Phase 5** — the native plugin suite from decision 3.
* **Phase 6** — `apps/kvim` with your config baked in.

Phases 1–3 deliver a working, Android-capable modal editor with LSP **without
any Lua at all**. That ordering is deliberate: it front-loads everything that
is definitely needed and defers the one decision I am least sure of.

## Amendment (same day): decision 2 was half-right

I claimed the Android fix was "fetch prebuilt static LSP binaries instead of
Mason's npm/pip/go shell-outs." While implementing it, the LSP work checked the
GitHub releases API rather than trusting my assumption, and found that **none of
the three servers you use — rust-analyzer, lua-language-server, texlab — publish
an `aarch64-linux-android` build at all.** They ship glibc/musl Linux, Darwin,
and Windows.

So the diagnosis in decision 2 stands (Mason fails because of the *toolchain
shell-outs*, not Lua) but the *remedy* is incomplete: there is nothing to
download. The registry now honestly reports `Unavailable` on Android rather than
fabricating a URL that would 404, and the real remedy is filed as
`kopitiam-cj0.9`. The likely answers are Termux's own package repo (which is not
GitHub releases), cross-compiling the servers ourselves, or — for rust-analyzer
specifically, since it is a Rust program — building it for the target and
possibly linking it straight into kvim.

This is worth stating plainly: the correction came from *checking*, not from
reasoning. The original claim was plausible and wrong.

## What would make this wrong

* **If you want a working editor sooner and don't care that the Lua VM is C**,
  reverse decision 1: take `mlua`, treat it as a permitted "optional
  integration," and skip Phase 4 almost entirely. This roughly halves the
  project. I did not choose it because CLAUDE.md's Pure Rust Core section is
  unusually emphatic ("Long-term ownership of the platform is more important
  than short-term convenience") and because you flagged Android, where a
  vendored C library is at least an irritant. But it is a close call and it is
  *your* call.
* **If you actually want your existing Lua plugins to keep working** (rather
  than being replaced by Rust equivalents), then decision 3 is wrong, the Lua
  VM becomes the critical path rather than a Phase 4, and it must be
  bug-for-bug compatible with LuaJIT — a substantially harder target than Lua
  5.1.
* **If `kvim` should genuinely read `~/.config/nvim/`** and act as a drop-in
  Neovim replacement on your existing setup, decision 5 is wrong.

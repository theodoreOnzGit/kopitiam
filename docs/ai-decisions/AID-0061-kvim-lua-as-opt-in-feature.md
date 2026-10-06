# AID-0061 — kvim's Lua config layer comes back as an opt-in `lua` cargo feature, off by default; released as 0.4.1

**Status:** Pending review
**Date:** 2026-10-06 (SGT)
**Issue:** gh-118 / `bd-8p3`
**Amends:** [AID-0060](AID-0060-kvim-drops-lua-config.md) (kvim drops its Lua config layer, 0.4.0)

## Context

Same day as AID-0060, after 0.4.0 went live, the maintainer refined the call:

> keep kopitiam-lua as a feature gated dependency

> for kvim, publish 0.4.1. off by default.

So the *direction* is again the maintainer's: the Lua layer is not deleted, it
is fenced off behind a cargo feature nobody gets unless they ask. This AID
records how that fence was built and the one SemVer trade-off it carries.

## What was done

* `src/luaconfig/` (excmd.rs, mod.rs, shim.rs, tests.rs — 1,800 lines)
  **restored verbatim** from `93f0f01^` (= `48b446c`, the last commit that had
  it), not rewritten. `lib.rs` declares it under `#[cfg(feature = "lua")]`.
* Every call site 93f0f01 removed is restored from the same commit, each one
  `#[cfg(feature = "lua")]`: `Config::lua_files` (+ its test),
  `Action::FeedKeys(String)` / `Action::LuaKeymap(usize)`, the `App::lua` field,
  `App::set_lua_runtime`, `App::set_startup_message`, `App::fire_lua_keymap` and
  its `handle_action` arm, `ui::bootstrap::apply_lua_config` (+ its test) and
  its two call sites in `ui::run`, the Lua section of `kvim --config-path`.
* `Cargo.toml`: `kopitiam-lua = { workspace = true, optional = true }` and
  `[features] default = []`, `lua = ["dep:kopitiam-lua"]`. The `dep:` prefix
  means there is **no** implicit `kopitiam-lua` feature — `lua` is the only
  public name for it.
* Default build is **0.4.0 exactly**: `kopitiam-lua` not compiled, `kvim --help`
  byte-identical (the Lua sentence is a second `const` that is `""` without the
  feature), `--config-path` still says kvim never reads or runs Lua (plus one
  new line pointing at `--features lua`). Default test count: **923 passed** —
  the same number 0.4.0 recorded.
* `--features lua`: **953 passed** = 923 + the 28 `luaconfig::tests` + the
  `lua_files` and `apply_lua_config` tests that 0.4.0 deleted.

## Decision 1 — the gated `Action` variants, and why that is accepted

`config::Action` is **not** `#[non_exhaustive]`. With `lua` on, it gains two
variants. Cargo features are unified across a build graph, so if crate A
`match`es `Action` exhaustively and crate B (anywhere in the same graph) turns
on `kopitiam-neovim/lua`, crate A stops compiling. That is precisely the
"features must be additive" hazard the Cargo book warns about, and kopitiam
itself has treated a new `Action` variant as breaking before (AID-0059 → 0.3.0).

**Why accept it anyway, as a 0.4.1 patch:**

1. **The default build is a strict superset of 0.4.0** — actually identical
   API. Nobody who leaves the feature off (that is everybody today) can be
   broken by 0.4.1. The SemVer promise is about the *default* surface, and that
   did not move.
2. **The hazard needs a third party to opt in.** It bites only when someone
   enables `lua` *and* something else in the same graph matches `Action`
   exhaustively. The one dependent in existence, `apps/cli`, only calls
   `Config::load` and `ui::run` (checked:
   `grep -rn kopitiam_neovim apps crates --include=*.rs`), and does not enable
   `lua`.
3. **The alternatives are worse for the maintainer's actual ask.**
   * *Variants unconditional* (always present, only produced under `lua`):
     adds variants to an exhaustive enum in the default build → breaking →
     **0.5.0**, contradicting "publish 0.4.1". And a default-build `Action`
     with two variants that nothing can produce is the dead-config-surface
     problem AID-0060 decision 2 removed.
   * *`#[non_exhaustive]` on `Action`*: the clean long-term fix, but adding it
     is itself breaking → 0.5.0. Same contradiction.
   * *Keep the Lua actions out of `Action` entirely* (e.g. a side table in
     `LuaRuntime` keyed by keymap, dispatched by `App` without an `Action`): a
     redesign of the shim, not a restore — and the brief was "recover, do not
     rewrite".

**Recommended follow-up, not done here:** at the next breaking kvim release
anyway, make `Action` `#[non_exhaustive]`. Then the gated variants stop being a
hazard at all.

## Decision 2 — `App::set_startup_message` is gated too

It is generic-looking (sets an Info statusline message), but its only caller is
the Lua startup note. Ungated it would be a new public method in the default
build — harmless SemVer-wise (additive), but it would make the default API a
proper superset rather than *identical* to 0.4.0, and with no caller it is dead
weight. So: Lua-only, like the rest.

## Decision 3 — `kopitiam-lua` goes back into `publish-kvim.sh`'s list

crates.io requires every dependency, **optional included**, to exist on the
registry at a matching version before it accepts `kopitiam-neovim`. It does:
`kopitiam-lua` 0.2.5 (published 2026-07-29), and
`git log --since=2026-07-29T00:09:01Z -- crates/kopitiam-lua` is empty, so per
CLAUDE.md "no new commits, no version bump" it **holds at 0.2.5** and the
script just prints "already published, skipping". It is listed anyway so the
kvim publish tree is complete if `kopitiam-lua` ever moves. `publish.sh`
unchanged (it already lists `kopitiam-lua`; only its comment says "nothing
depends on it", now stale → corrected).

## Decision 4 — `apps/cli` requirement stays `"0.4.0"`

`"0.4.0"` is a caret requirement and already admits 0.4.1; the CLI does not
enable `lua`. Nothing to change, so nothing changed.

## What would make this wrong

* **If someone enables `lua` alongside an exhaustive `match` on `Action`.**
  Then 0.4.1 breaks their build, and the honest number would have been 0.5.0.
  Mitigated in docs (README "one caveat hor", rustdoc on both variants), not
  in code.
* **If "off by default" was meant at runtime, not compile time** — e.g. Lua
  always compiled in but only run when a config flag says so. The maintainer
  said "feature gated dependency", which in cargo means a feature; read that
  way.
* **If the maintainer wanted `kopitiam-lua` kept out of publish-kvim.sh.**
  Removing it would not break the publish today (0.2.5 is live), only a
  future one after a `kopitiam-lua` bump.

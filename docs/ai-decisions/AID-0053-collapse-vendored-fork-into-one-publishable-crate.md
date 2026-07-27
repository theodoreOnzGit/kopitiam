# AID-0053: Collapse a vendored multi-crate fork into ONE publishable crate, to avoid squatting upstream sub-crate names on crates.io

* **Status:** Pending review
* **Date:** 2026-07-27
* **Decided by:** AI (Claude), maintainer absent
* **Bead:** —
* **Scope:** the two whole-project forks KOPITIAM ships — the `rmux` fork
  (`crates/kmux/`) and the `beads-rs` fork (`crates/kopi-beans/`). This is the
  publish-time packaging convention for **any** future vendored multi-crate
  fork; it does not change the fork-vs-clean-room rules of AID-0006 or the
  attribution rules in `docs/ACKNOWLEDGEMENTS.md`.

## The brief

Both forks were taken *whole*: upstream is a Cargo **workspace of many crates**,
and each was originally vendored with its sub-crates nested under
`crates/<fork>/crates/`, keeping the upstream crate names so diffs against
upstream stayed readable (the AID-0006 decision, and the same for beads-rs).
That layout is correct for a `publish = false` in-tree fork. It becomes a
**problem the moment the fork should go on crates.io**: publishing the workspace
as-is would either (a) require publishing every upstream sub-crate under its
**original name** (`rmux-core`, `beads-core`, …) — squatting names that belong to
the upstream authors and that they may want to publish themselves — or (b)
rename every sub-crate to a `kopitiam-*` prefix, multiplying the published
package count and the version surface for what the user installs as **one
binary** (`kmux`, `bn`).

## The decision

**Collapse each multi-crate fork into ONE self-contained, publishable crate.**
Fold every sub-crate the shipped binary actually needs into the top crate as an
**intra-crate module**, and publish exactly that one package.

Concretely, the convention both forks followed:

1. **One package, sub-crates become modules.** Each needed sub-crate's `src/`
   becomes a module of the single crate (`rmux-core` → `crate::core`,
   `beads-core` → `crate::core`, …); each sub-crate's `lib.rs` becomes that
   module's `mod.rs`. Cross-crate paths (`rmux_core::X`, `beads_core::X`) are
   rewritten to the intra-crate form (`crate::core::X`).
2. **Union the manifests.** The sub-crate `Cargo.toml` dependency sets are
   merged into the one crate's manifest, **carrying upstream's version pins
   verbatim**. Sub-crate build scripts (e.g. rmux-server's tunnel-preset codegen)
   merge into the single `build.rs`.
3. **Package name is new; lib name stays upstream.** The published **package** is
   renamed to a KOPITIAM-namespaced, non-squatting name (`kmux`, `kopi-beans`);
   where it helps future merges, the `[lib] name` keeps the upstream identifier
   (`beads_rs`). The point is that **no crates.io package is published under a
   name the upstream authors own.**
4. **Drop what the binary never links.** Sub-crates outside the shipped binary's
   graph are dropped, not folded (`ratatui-rmux`, `rmux-render-core` for kmux;
   the `beads-http` dev transport, the `fuzz` crate, and the `stateright` /
   `beads_stateright_models` harness for kopi-beans). Inline `#[cfg(test)]` unit
   tests inside folded modules are kept; separate `tests/` integration suites and
   upstream CI/packaging scripts are not.
5. **The NOTICE records the collapse**, in addition to the fork — which module
   each sub-crate became, what was dropped, and that upstream authorship and
   license are unchanged.

## Why this is the maintainer's call, and why it went this way

Whether KOPITIAM publishes these forks at all — and under what package identity —
is a project-identity and licensing decision, not a mechanical one. The
alternatives were:

* **(a) Publish the workspace as-is**, each sub-crate under its upstream name.
  Rejected: it squats names belonging to The RMUX Authors and Darin Kishore on a
  public registry, which is exactly the kind of thing the attribution discipline
  in `ACKNOWLEDGEMENTS.md` exists to avoid — a permissive license lets you *reuse*
  the code, it does not entitle you to the authors' registry namespace.
* **(b) Rename every sub-crate `kopitiam-*` and publish ~10 packages per fork.**
  Rejected: it inflates the published surface and the version-bump burden for a
  single-binary tool, and every one of those packages would be a
  KOPITIAM-branded copy of someone else's crate, which is louder about the fork
  than a single honestly-labelled crate is.
* **(c) Collapse to one crate** (chosen): one honestly-named package per binary,
  no squatting, the whole fork's provenance in one NOTICE, and a smaller version
  surface. The cost is that diffs against upstream get harder — the module
  rewrite means paths no longer line up file-for-file — which is the readability
  benefit the nested layout was originally chosen for (AID-0006). That trade is
  worth it **at publish time**: readable upstream diffs matter most while
  actively porting, and both forks are now feature-stable ports, not moving
  targets.

For **kopi-beans specifically**, the collapse is one of *two* preconditions for
publishability; the other — becoming fully pure-Rust by dropping `rusqlite`'s
bundled SQLite and porting `git2`/libgit2 to `gitoxide` — is an application of
the existing Pure Rust Core principle (cf. AID-0052's substitute-don't-link
stance) rather than a new decision, and is recorded in `ACKNOWLEDGEMENTS.md`
rather than warranting its own AID. The one thing there that *is* a live gap
worth flagging is that `gix` 0.86 has no high-level push, so `bn`'s push is gated
pending a send-pack shim — noted at the point of use and in the release notes,
not buried.

## What would make this wrong

* **If upstream becomes a moving target again** — e.g. KOPITIAM wants to track
  ongoing rmux/beads-rs development and merge upstream regularly — the collapse's
  loss of file-for-file diffability becomes a real recurring tax, and keeping the
  nested `crates/<fork>/crates/` layout (published under `kopitiam-*` names, or
  simply left `publish = false`) would be the better trade. The collapse bets
  that these forks are stable ports.
* **If a sub-crate that was dropped turns out to be needed** (e.g. a future
  feature links `rmux-render-core`), it has to be folded back in rather than
  re-added as a path dependency, since the single-crate shape is now the
  published contract.
* **If the maintainer would rather not publish these forks to crates.io at all**
  — keeping them in-tree, `publish = false` — then the whole collapse was
  unnecessary and the readable nested layout should have been preserved. The
  collapse assumes the crates.io debut is wanted.

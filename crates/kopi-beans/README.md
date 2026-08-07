# kopi-beans (`bn`)

a distributed, git-backed work-item tracker for agent swarms. conflicts are
impossible. the binary is **`bn`**.

kopi-beans is a **fork of [beads-rs](https://github.com/delightful-ai/beads-rs)**
(MIT, © 2025 Darin Kishore), itself a rust redesign of
[beads](https://github.com/steveyegge/beads). it adds Windows and Termux/Android
support, is shipped as part of [KOPITIAM](https://github.com/theodoreOnzGit/kopitiam),
and is relicensed **AGPL-3.0-only**. see `NOTICE` for full provenance and the
list of changes.

> **kopi-beans is a separate tool from beads-rs.** `bn` and `bd` can be
> installed side by side. `bn` never reads, writes, upgrades, or removes the
> `bd` binary, and it will not touch beads-rs's editor hooks.

## install

```bash
cargo install kopi-beans          # binary: bn
```

crates.io is the only distribution channel — there are no prebuilt release
binaries, no install script, and no nix flake for this fork.

## quick start

in any git repo:

```bash
bn init
bn setup claude # or cursor or aider
```

then, ask your coding agent to run `bn onboard`, and you're done!

## why

beads lets agents coordinate work and defer problems to the next session with a
fresh context window. 50 agents on one codebase need to see the same task list -
who's working on what, what's blocked, what's done.

the go version works for one human with occasional sync. this lineage is
designed for swarms: one daemon per machine shares state across all clones
instantly. no sqlite, everything lives in git on `refs/heads/beads/store`.

this means agents never push to main when updating beads. no merge conflicts on
your code branches. it just works, and gets out of the way.

## status

alpha. designed for reliability first.

## differences from beads-go

a **drop-in replacement** for core workflows. the main difference is the sync
model - agents never need to run `bn sync`, beads can't have merge conflicts,
and it works across machines automatically.

**what's missing:**

| feature | status |
|---------|--------|
| agent mail (real-time multi-agent) | not yet |
| multi-repo state sharing | not yet |
| jira integration | not yet |
| doctor/repair commands | not yet |
| templates | not yet |
| compaction/decay | not yet |
| self-upgrade (`bn upgrade`) | removed on purpose - see [upgrade](#upgrade) |

if you need agent mail or multi-repo, use
[the original](https://github.com/steveyegge/beads).

## migration path

```bash
bn migrate from-go --input .beads/issues.jsonl --dry-run
# if that looks good, run
bn migrate from-go --input .beads/issues.jsonl
# and you're good!
```

`migrate` is hidden from the top-level help; run `bn migrate --help` for the
full set (`detect`, `to`, `from-go`).

## technical details

**requirements:**
- git repo with an `origin` remote (recommended; local-only works too)
- linux, macos, windows, and termux/android

**where data lives:**
- canonical state: `refs/heads/beads/store`
- bounded backup refs: `refs/beads/backup/*` (latest 64 retained)
- files: `state.jsonl`, `tombstones.jsonl`, `deps.jsonl`, `meta.json`
- daemon socket: `$XDG_RUNTIME_DIR/beads/daemon.sock` or `~/.beads/daemon.sock` or `/tmp/beads-$uid/daemon.sock`

these paths and file names are **deliberately identical to beads-rs's** — it is
the same on-disk format, not a coincidence.

**sync model:**
- cli auto-starts a local daemon on demand
- mutations are debounced and pushed in the background
- `bn sync` is just "wait for flush", not a workflow step
- backup ref maintenance is best-effort under lock contention and uses age/PID-aware stale lock cleanup

## editor integration

```bash
bn setup claude
bn setup cursor
bn setup aider
```

these register a `bn prime` hook. if you also run beads-rs, `bn` will report a
pre-existing `bd prime` hook but will never edit or delete it — removing another
tool's configuration is your call, not `bn`'s.

## upgrade

```bash
cargo install kopi-beans
```

`bn` does **not** self-upgrade, and `bn upgrade` installs nothing — it prints
the line above and exits non-zero.

self-upgrade was removed deliberately. the implementation inherited from
beads-rs downloaded **beads-rs's** GitHub releases and installed them over a
binary named **`bd`**, so on a machine running both trackers `bn upgrade` would
have overwritten the other tool. kopi-beans also publishes no release artifacts
of its own, so there was nothing to re-point it at. auto-upgrade is likewise a
no-op.

## config

kopi-beans reads a single config file:

```
~/.config/beads-rs/config.toml
```

override the location with `BD_CONFIG_DIR`. the `BD_*` environment variables and
the config directory name are inherited from beads-rs and kept unchanged for
compatibility with existing setups.

the `auto_upgrade` key is still parsed but has no effect (see
[upgrade](#upgrade)).

## docs

- `NOTICE` - upstream attribution and the full list of fork changes
- `bn --help` / `bn <cmd> --help` - command reference
- `bn prime` - the workflow context `bn` hands to coding agents

## license

**AGPL-3.0-only** (see `LICENSE`).

upstream beads-rs is MIT and remains available from its author under that
licence; its unmodified licence text is kept alongside this file as
`LICENSE-MIT`. relicensing this fork does not relicense upstream beads-rs.

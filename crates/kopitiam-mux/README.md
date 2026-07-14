# kopitiam-mux (`kmux`)

KOPITIAM's terminal multiplexer. A **fork of [rmux](https://github.com/helvesec/rmux)**
(© The RMUX Authors, MIT OR Apache-2.0), relicensed AGPL-3.0-only as part of
KOPITIAM, forked to add **Android/Termux** support alongside Linux, macOS and
Windows.

See [`NOTICE`](NOTICE) for the full attribution and the list of changes, and
`docs/ai-decisions/AID-0006` for why a fork was necessary rather than a patch.

## Layout

This crate is a **nested workspace**. The outer KOPITIAM workspace globs
`crates/*` and therefore sees exactly one member — `kopitiam-mux` — while the
twelve forked sub-crates live under `crates/kopitiam-mux/crates/` and are pulled
in as path dependencies:

```
crates/kopitiam-mux/
  Cargo.toml          # the only crate the root workspace glob sees
  src/                # the kmux + kmux-daemon binaries
  crates/
    rmux-types/  rmux-proto/  rmux-core/   rmux-os/
    rmux-ipc/    rmux-pty/    rmux-client/ rmux-server/
    rmux-sdk/    rmux-render-core/  rmux-web-crypto/  ratatui-rmux/
```

Sub-crates **keep their upstream names** so that diffs against upstream stay
readable for the next decade, but carry a `-kopitiam` version suffix and
`publish = false` — a modified `rmux-os` can never be mistaken for the real one.

## Build

```sh
cargo build --release -p kopitiam-mux      # produces target/release/kmux
cargo test  --release -p kopitiam-mux
```

## Android

Type-checks clean on `aarch64-linux-android`, `armv7-linux-androideabi` and
`x86_64-linux-android`:

```sh
cargo check --release -p kopitiam-mux --target aarch64-linux-android
```

**It has not yet been run on a device.** Type-checking proves the cfg gates
resolve and the code compiles for Bionic; it does not prove the PTY opens or the
daemon survives Android's process lifecycle. See the beads for the on-device
work.

The canonical write-up of every Android decision — which cfg gates were widened,
which were deliberately *not*, and why Termux detection keys on `$PREFIX` rather
than `cfg!(target_os = "android")` — is the module documentation of
[`rmux_os::runtime_dir`](crates/rmux-os/src/runtime_dir.rs). Read it before
touching a `cfg` gate.

//! `bn upgrade` support.
//!
//! # Why this module installs nothing
//!
//! kopi-beans inherited beads-rs's self-upgrade machinery verbatim during the
//! fork, and that machinery was still pointed at **beads-rs**, not at
//! kopi-beans (kopitiam#14):
//!
//! - it fetched `https://api.github.com/repos/delightful-ai/beads-rs/releases/latest`,
//! - it unpacked the archive member named `bd`,
//! - it ran `cargo install --git https://github.com/delightful-ai/beads-rs.git`,
//! - and it resolved its install target by looking for a binary called `bd` on
//!   `PATH`, falling back to `<install-dir>/bd`.
//!
//! On a machine with both trackers installed — the exact configuration the bug
//! was reported from — `bn upgrade` would therefore have located the user's
//! **beads-rs `bd`** binary and overwritten it with an upstream **beads-rs**
//! build. That is a destructive cross-tool action, so the whole install path
//! has been removed rather than merely re-pointed.
//!
//! It was not re-pointed at kopi-beans because there is nothing to point it at:
//! the `theodoreOnzGit/kopitiam` repository publishes **no GitHub releases**
//! (`GET /repos/theodoreOnzGit/kopitiam/releases` returns `[]`, and
//! `.../releases/latest` returns HTTP 404), so there are no prebuilt archives
//! and no release tag to compare against. kopi-beans is distributed solely as a
//! crates.io package.
//!
//! `bn upgrade` therefore fails loudly with [`UPGRADE_UNSUPPORTED_MESSAGE`],
//! which tells the user the one command that does upgrade `bn`:
//!
//! ```text
//! cargo install kopi-beans
//! ```
//!
//! **Invariant:** no code path in this module opens, copies, renames, or
//! otherwise writes any file — least of all one named `bd`. The
//! `upgrade_module_has_no_binary_install_path` test below enforces that against
//! this file's own source so the beads-rs targeting cannot silently return.

use std::path::PathBuf;

use serde::Serialize;

use crate::OpError;
use crate::config::Config;
use crate::{Error, Result};

/// Message shown when a user runs `bn upgrade`.
///
/// kopi-beans ships no release artifacts, so self-upgrade is not available;
/// this names the supported upgrade path instead of silently doing nothing (or,
/// as before, silently doing something destructive).
pub const UPGRADE_UNSUPPORTED_MESSAGE: &str = concat!(
    "`bn upgrade` cannot install anything: kopi-beans publishes no prebuilt ",
    "release binaries, only the crates.io package. Upgrade `bn` with:\n",
    "\n",
    "    cargo install kopi-beans\n",
    "\n",
    "(Self-upgrade was removed in kopitiam#14: it targeted the unrelated ",
    "beads-rs `bd` binary and its upstream releases, so on a machine with both ",
    "trackers installed it would have overwritten `bd`.)"
);

/// Outcome of an upgrade attempt.
///
/// Retained because it is the host half of the CLI's upgrade protocol
/// (`cli_surface::backend::UpgradeOutcome`). No value of this type is currently
/// produced: [`run_upgrade`] always fails. See the module docs.
#[derive(Debug, Clone)]
pub struct UpgradeOutcome {
    /// Whether a new binary was installed.
    pub updated: bool,
    /// Version `bn` was running before the attempt.
    pub from_version: String,
    /// Version `bn` would have been upgraded to, if known.
    pub to_version: Option<String>,
    /// Path the new binary would have been written to.
    pub install_path: PathBuf,
    /// How the upgrade was performed.
    pub method: UpgradeMethod,
}

/// How an upgrade was carried out.
///
/// `Prebuilt` and `Cargo` are unreachable today — kept so the host and CLI
/// halves of the upgrade protocol stay 1:1 if release artifacts are ever
/// published.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UpgradeMethod {
    /// Installed from a downloaded prebuilt archive.
    Prebuilt,
    /// Installed by building from source with `cargo install`.
    Cargo,
    /// Nothing was installed.
    None,
}

/// Background auto-upgrade hook, invoked once per CLI startup.
///
/// This is deliberately a no-op. `bn` has no self-upgrade path (see the module
/// docs), so there is nothing to check for in the background, and the previous
/// implementation spawned `bn upgrade --background`, which — before this
/// change — could silently replace a `bd` binary with no user interaction at
/// all. That is the worst version of the kopitiam#14 bug and it must not be
/// reachable.
pub fn maybe_spawn_auto_upgrade() {}

/// Handle `bn upgrade`.
///
/// Always fails with [`UPGRADE_UNSUPPORTED_MESSAGE`]. kopi-beans is installed
/// and upgraded through crates.io (`cargo install kopi-beans`); it publishes no
/// release artifacts for `bn` to fetch, and it must never fetch or install
/// beads-rs's. See the module docs for the full rationale.
///
/// `background` is accepted (the hidden `--background` flag still parses) but
/// ignored: there is no longer a background upgrade to perform.
pub fn run_upgrade(_config: Config, _background: bool) -> Result<UpgradeOutcome> {
    Err(upgrade_error(UPGRADE_UNSUPPORTED_MESSAGE.to_string()))
}

fn upgrade_error(reason: String) -> Error {
    Error::Op(OpError::ValidationFailed {
        field: "upgrade".into(),
        reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err_text(err: Error) -> String {
        match err {
            Error::Op(OpError::ValidationFailed { reason, .. }) => reason,
            other => panic!("expected a validation error, got {other:?}"),
        }
    }

    #[test]
    fn run_upgrade_refuses_and_points_at_cargo_install() {
        let err = run_upgrade(Config::default(), false).expect_err("upgrade must not succeed");
        let text = err_text(err);
        assert!(text.contains("cargo install kopi-beans"), "got: {text}");
        assert!(text.contains("kopi-beans"), "got: {text}");
    }

    #[test]
    fn run_upgrade_refuses_in_background_mode_too() {
        let err =
            run_upgrade(Config::default(), true).expect_err("background upgrade must not succeed");
        assert!(err_text(err).contains("cargo install kopi-beans"));
    }

    /// Regression guard for kopitiam#14.
    ///
    /// The reported hazard was not the wording of the help text but the code
    /// underneath it: `bn upgrade` resolved its install target by name and that
    /// name was `bd`, so it could overwrite an unrelated beads-rs install. This
    /// test reads this module's own source and fails if any of the removed
    /// machinery — a `bd`-named target, the beads-rs release endpoint, or any
    /// filesystem write primitive — reappears.
    ///
    /// Two slices of the file are excluded so the guard does not match itself:
    /// the `#[cfg(test)]` module (which contains the needles as string
    /// literals) and comment lines (the module docs quote the removed URLs on
    /// purpose, to record what was there and why it went).
    #[test]
    fn upgrade_module_has_no_binary_install_path() {
        let source = include_str!("upgrade.rs");
        let code: String = source
            .split("#[cfg(test)]")
            .next()
            .expect("source always has a pre-test half")
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");

        for forbidden in [
            "\"bd\"",
            "delightful-ai",
            "beads-rs.git",
            "releases/latest",
            "std::fs",
            "fs::copy",
            "fs::write",
            "fs::rename",
            "persist(",
            "Command::new",
        ] {
            assert!(
                !code.contains(forbidden),
                "src/upgrade.rs must not contain {forbidden:?}: the upgrade path \
                 must stay incapable of installing or overwriting any binary \
                 (kopitiam#14)"
            );
        }
    }
}

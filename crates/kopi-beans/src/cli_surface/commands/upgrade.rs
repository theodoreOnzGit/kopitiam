//! `bn upgrade` - report how to upgrade `bn`.
//!
//! The host side no longer installs anything: kopi-beans ships no release
//! artifacts and the inherited beads-rs self-upgrade targeted the wrong binary
//! entirely (see `crate::upgrade` and kopitiam#14). `run_upgrade` therefore
//! always fails with a message naming `cargo install kopi-beans`, and the
//! rendering below is kept only so the CLI/host protocol stays whole.

use clap::Args;
use serde::Serialize;

use super::CommandError;
use crate::cli_surface::backend::{CliHostBackend, UpgradeMethod, UpgradeOutcome, UpgradeRequest};
use crate::cli_surface::render::{print_json, print_line};

#[derive(Args, Debug)]
pub struct UpgradeArgs {
    /// Run upgrade in the background (internal).
    #[arg(long, hide = true, default_value_t = false)]
    pub background: bool,
}

#[derive(Serialize)]
struct UpgradeJson<'a> {
    updated: bool,
    from_version: &'a str,
    to_version: Option<&'a str>,
    install_path: &'a str,
    method: &'a str,
}

pub fn handle<B>(json: bool, background: bool, backend: &B) -> std::result::Result<(), B::Error>
where
    B: CliHostBackend,
    B::Error: From<CommandError>,
{
    let outcome = backend.run_upgrade(UpgradeRequest { background })?;
    if json {
        let payload = UpgradeJson {
            updated: outcome.updated,
            from_version: &outcome.from_version,
            to_version: outcome.to_version.as_deref(),
            install_path: outcome.install_path.to_str().unwrap_or(""),
            method: method_str(outcome.method),
        };
        print_json(&payload).map_err(|err| B::Error::from(CommandError::from(err)))?;
        return Ok(());
    }

    if background {
        return Ok(());
    }

    render_human(&outcome).map_err(|err| B::Error::from(CommandError::from(err)))
}

fn render_human(outcome: &UpgradeOutcome) -> crate::cli_surface::Result<()> {
    if !outcome.updated {
        return print_line(&format!(
            "bn is up to date (version {}).",
            outcome.from_version
        ));
    }

    print_line(&format!(
        "Upgraded bn from {} to {} using {}.",
        outcome.from_version,
        outcome.to_version.as_deref().unwrap_or("unknown"),
        method_str(outcome.method),
    ))?;
    print_line(&format!("Installed: {}", outcome.install_path.display()))
}

fn method_str(method: UpgradeMethod) -> &'static str {
    match method {
        UpgradeMethod::Prebuilt => "prebuilt binary",
        UpgradeMethod::Cargo => "cargo build",
        UpgradeMethod::None => "none",
    }
}

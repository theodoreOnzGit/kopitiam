//! The `bn` subcommand: an arm's-length subprocess passthrough to `kopi-beans`
//! (issue #30).
//!
//! # Why a passthrough, not a dependency
//!
//! `kopi-beans` is KOPITIAM's task/bead ledger, shipped as its own `bn` binary.
//! We deliberately do NOT add `kopi-beans` as a Cargo dependency: `kopitiam`
//! stays pure-Rust and self-contained, and the ledger tool evolves on its own
//! release cadence. Instead, `kopitiam bn <args...>` finds the `bn` binary on
//! `PATH`, spawns it with **every** argument after `bn` forwarded verbatim,
//! inherits stdin/stdout/stderr, and exits with the child's exact exit code.
//! From a caller's point of view `kopitiam bn create "x" -t task` is
//! indistinguishable from running `bn create "x" -t task` directly.
//!
//! This is agent-safe: `bn` itself is non-interactive, so the passthrough is
//! too. It is NOT flagged interactive in `kopitiam_skill.md`.
//!
//! # Thin-client discipline
//!
//! There is no business logic here — this module only assembles a `Command`,
//! runs it, and translates "binary not found" into an actionable install hint.
//! The `Args`/`run` split keeps the argument-collection and the not-installed
//! message unit-testable without a real `bn` on `PATH`.

use std::process::{Command, ExitCode, Stdio};

use clap::Args;

/// The name of the `kopi-beans` binary we shell out to. It lives on `PATH`
/// (e.g. after `cargo install kopi-beans`), not inside this crate.
const BN_BINARY: &str = "bn";

/// Shown on stderr when `bn` is not on `PATH`. Names the exact command that
/// fixes it, so a stranded user (or agent) can self-serve.
const NOT_INSTALLED_MSG: &str = "kopi-beans not installed — run: cargo install kopi-beans";

/// Options for `kopitiam bn`: a raw, verbatim tail of arguments.
///
/// `trailing_var_arg` + `allow_hyphen_values` together tell clap to stop
/// interpreting anything after `bn` and hand it all through — including
/// `-t`/`--flag` style options and their values — so `bn`'s own argument
/// grammar (not clap's) is the only one that ever parses them.
#[derive(Args, Debug)]
pub struct BnArgs {
    /// Every argument after `bn`, forwarded to the `bn` binary untouched.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<String>,
}

/// Build the `bn` subprocess command with `args` forwarded verbatim and all
/// three standard streams inherited.
///
/// Split out from [`run`] so a test can assert the program name and the
/// forwarded argument vector without a real `bn` on `PATH`.
fn build_command(args: &[String]) -> Command {
    let mut cmd = Command::new(BN_BINARY);
    cmd.args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    cmd
}

/// Runs `kopitiam bn <args...>`: spawn `bn` on `PATH`, inherit all three
/// streams, and exit with the child's exit code.
///
/// On a normal run this **never returns** — it calls [`std::process::exit`]
/// with the child's code so `kopitiam bn` composes in shell scripts exactly
/// like `bn` itself. The only value it hands back is [`ExitCode::FAILURE`] when
/// `bn` is not installed (after printing the install hint), so `main` can
/// surface a non-zero status.
pub fn run(args: BnArgs) -> anyhow::Result<ExitCode> {
    let mut cmd = build_command(&args.args);

    match cmd.status() {
        Ok(status) => {
            // Forward the child's exact exit code. `.code()` is `None` only when
            // the child was killed by a signal (Unix); map that to a non-zero
            // failure so we never masquerade a killed child as success.
            std::process::exit(status.code().unwrap_or(1));
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("{NOT_INSTALLED_MSG}");
            Ok(ExitCode::FAILURE)
        }
        Err(err) => Err(anyhow::Error::new(err)
            .context(format!("failed to run the `{BN_BINARY}` binary"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every argument — plain, quoted, and hyphen-flag alike — is forwarded to
    /// `bn` verbatim and in order, and the program spawned is `bn`.
    #[test]
    fn build_command_forwards_all_args_verbatim() {
        let args = vec![
            "create".to_string(),
            "x".to_string(),
            "-t".to_string(),
            "task".to_string(),
        ];
        let cmd = build_command(&args);

        assert_eq!(cmd.get_program(), BN_BINARY);
        let forwarded: Vec<&str> = cmd
            .get_args()
            .map(|a| a.to_str().unwrap())
            .collect();
        assert_eq!(forwarded, ["create", "x", "-t", "task"]);
    }

    /// An empty tail (`kopitiam bn` with nothing after it) still spawns `bn`
    /// with no extra args — `bn`'s own no-arg behaviour (its help/usage) then
    /// takes over.
    #[test]
    fn build_command_with_no_args_spawns_bare_bn() {
        let cmd = build_command(&[]);
        assert_eq!(cmd.get_program(), BN_BINARY);
        assert_eq!(cmd.get_args().count(), 0);
    }

    /// The not-installed message must name the exact `cargo install` command,
    /// so a user or agent without `bn` on `PATH` can fix it without guessing.
    #[test]
    fn not_installed_message_names_the_install_command() {
        assert!(NOT_INSTALLED_MSG.contains("cargo install kopi-beans"));
        assert!(NOT_INSTALLED_MSG.contains("kopi-beans not installed"));
    }
}

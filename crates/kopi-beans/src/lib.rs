// NOTE: this crate is a SINGLE-crate collapse of the former multi-crate
// `kopitiam-bds` fork of beads-rs. The nine former sub-crates are now the
// modules declared below (core/api/bootstrap/surface/git/daemon_core/daemon/
// cli_surface + the macros module). A crate-wide `#![forbid(unsafe_code)]`
// is therefore NOT set here: the `surface` and `daemon` modules carry the
// Windows/Termux port's `unsafe` FFI (Win32 process + AF_UNIX plumbing). The
// pure-Rust modules keep their own module-scoped `#![forbid(unsafe_code)]`.
#![allow(clippy::result_large_err)]

// ===== Folded-in beads-rs sub-crates, now intra-crate modules =====
// `enum_str!` is `#[macro_export]` in `macros`, so it is reachable crate-wide as
// `crate::enum_str!` and to downstream consumers as `beads_rs::enum_str!`.
mod macros;
pub mod api;
pub mod bootstrap;
pub mod cli_surface;
pub mod core;
pub mod daemon;
pub mod daemon_core;
pub mod git;
pub mod surface;

// ===== Original kopitiam-bds top-crate modules =====
#[cfg(feature = "cli")]
pub mod cli;
pub mod config;
pub mod error;
#[cfg(feature = "model-testing")]
pub mod model;
pub mod paths;
mod repo;
pub(crate) mod store_admin;
mod telemetry;
mod upgrade;

pub use crate::daemon::compat;
pub use error::{Effect, Error, OpError, Transience};
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(feature = "cli")]
/// Thin orchestration shim for the `bn` binary.
///
/// Entry-point binaries should stay as minimal wiring while command behavior
/// lives behind crate boundaries.
pub fn run_cli_entrypoint(cli: cli::Cli) -> i32 {
    let is_daemon = matches!(
        cli.command,
        cli::Command::Daemon {
            cmd: cli::DaemonCmd::Run
        }
    );
    let _telemetry_guard = init_cli_tracing(cli.verbose, is_daemon);

    let command = cli::command_name(&cli.command);
    let span = tracing::info_span!(
        "cli_command",
        command = %command,
        repo = ?cli.repo
    );
    let _guard = span.enter();

    if let Err(err) = cli::run(cli) {
        tracing::error!("error: {}", err);
        return 1;
    }

    0
}

#[cfg(feature = "cli")]
fn init_cli_tracing(verbose: u8, is_daemon: bool) -> telemetry::TelemetryGuard {
    let cfg = match config::load() {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("config load failed, using defaults: {err}");
            let mut cfg = config::Config::default();
            config::apply_env_overrides(&mut cfg);
            cfg
        }
    };

    // Initialize path overrides from config before any IPC/daemon operations.
    paths::init_from_config(&cfg.paths);

    let mut logging = cfg.logging;
    if is_daemon {
        telemetry::apply_daemon_logging_defaults(&mut logging);
    }
    let telemetry_cfg = telemetry::TelemetryConfig::new(verbose, logging);
    telemetry::init(telemetry_cfg)
}

/// Stable wrapper for daemon-run entrypoint so CLI code doesn't import daemon internals directly.
pub fn run_daemon_command() -> Result<()> {
    let config = config::load_or_init();
    paths::init_from_config(&config.paths);
    let _socket_dir = crate::surface::ipc::ensure_socket_dir()?;

    let actor = daemon_actor_from_config(&config)?;
    let layout = daemon_layout_from_paths();
    let runtime = daemon_runtime_config_from_config(&config);
    Ok(crate::daemon::run_daemon(actor, layout, runtime)?)
}

pub(crate) fn daemon_layout_from_paths() -> crate::daemon::layout::DaemonLayout {
    crate::daemon::layout::DaemonLayout::new(
        paths::data_dir(),
        crate::surface::ipc::socket_path(),
        paths::log_dir(),
    )
}

pub(crate) fn daemon_runtime_config_from_config(
    config: &config::Config,
) -> crate::daemon::config::DaemonRuntimeConfig {
    crate::daemon::config::daemon_runtime_from_config(config)
}

fn daemon_actor_from_config(config: &config::Config) -> Result<crate::core::ActorId> {
    match config.defaults.actor.clone() {
        Some(actor) => Ok(actor),
        None => {
            let username = whoami::username();
            let hostname = whoami::fallible::hostname().unwrap_or_else(|_| "unknown".into());
            let default_actor = format!("{username}@{hostname}");
            Ok(crate::core::ActorId::new(default_actor)?)
        }
    }
}

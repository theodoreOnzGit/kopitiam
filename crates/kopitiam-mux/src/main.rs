#![deny(missing_docs)]

//! `kmux` — KOPITIAM's terminal multiplexer.
//!
//! # This crate is a fork of rmux
//!
//! `kopitiam-mux` is a **fork of [rmux](https://github.com/helvesec/rmux)**, not
//! an independent implementation and not a clean-room reimplementation. The
//! overwhelming majority of the code in this crate and in its nested sub-crates
//! (`crates/rmux-*`, `crates/ratatui-rmux`) was **written by the RMUX Authors**
//! and is reused here directly.
//!
//! * **Upstream:** <https://github.com/helvesec/rmux>
//! * **Upstream copyright:** "The RMUX Authors"
//! * **Upstream license:** MIT OR Apache-2.0. Both license texts travel with
//!   this fork, unmodified, as `LICENSE-MIT` and `LICENSE-APACHE` in this
//!   crate's directory, and the upstream copyright notices are retained.
//! * **This fork's license:** **AGPL-3.0-only**, like everything else in
//!   KOPITIAM. A permissive upstream may be absorbed into an AGPLv3 work
//!   provided its notices travel with the code, which is what the two license
//!   files above are for. Relicensing the *fork* does not and cannot relicense
//!   upstream rmux.
//!
//! # Why it was forked
//!
//! **rmux does not run on Android**, and KOPITIAM wants a terminal multiplexer
//! that runs everywhere it does — Android (via Termux) as well as Linux, macOS
//! and Windows.
//!
//! The gaps were not confined to one crate, which is what made a surgical patch
//! impossible and a fork necessary (see `docs/ai-decisions/AID-0006`):
//!
//! 1. **`target_os = "linux"` cfg gates silently exclude Android.** Android runs
//!    a Linux kernel but reports `target_os = "android"`, so a bare `linux` gate
//!    does not fail to compile on Android — the code simply *vanishes*. Gates
//!    guarding facilities Bionic genuinely has (`/proc`, `eventfd`, `setsid`,
//!    `close_range`, the Linux PTY backend) were widened to
//!    `any(target_os = "linux", target_os = "android")`. Gates guarding things
//!    Android should *not* have (abstract unix sockets) were deliberately left
//!    alone, so Android takes the portable filesystem-socket path instead.
//! 2. **Hardcoded `/tmp` and `/var/run`.** Termux is not FHS — its root is
//!    `/data/data/com.termux/files/` and it has no usable `/tmp`. All runtime
//!    paths now go through a single resolver, `rmux_os::runtime_dir`.
//!
//! `rmux_os::runtime_dir`'s module documentation is the canonical write-up of
//! every Android-specific decision in this fork. Read it before changing any
//! `cfg` gate here.
//!
//! # What changed from upstream
//!
//! Deliberately kept small, so upstream fixes stay mergeable:
//!
//! * The binaries are `kmux` / `kmux-daemon`, not `rmux` / `rmux-daemon`.
//! * Sub-crates keep their upstream names (`rmux-os`, `rmux-pty`, ...) so that
//!   diffs against upstream stay readable for the next decade. They carry a
//!   `-kopitiam` version suffix and `publish = false`, so a modified `rmux-os`
//!   can never be mistaken for the real one.
//! * `rmux_os::runtime_dir` (new) and `rmux_os::host`'s binary-name constants
//!   (new) are the only substantive additions.
//!
//! # The binary
//!
//! It owns two entrypoints:
//! - the public CLI that speaks the detached `rmux-proto` request/response API
//!   through `rmux-client`, and
//! - the hidden internal daemon mode used by tmux-style start-server commands.
//!
//! Optimized package builds can alternatively enable `tiny-cli`, making this
//! public binary a small dispatcher for hot detached commands while complex
//! commands exec the private full `rmux` helper installed under libexec.

#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
mod cli;
#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
mod cli_args;
#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
mod cli_response;
mod client_terminal;
#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
mod os_string;
#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
mod process_locale;
#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
mod server_runtime;
#[cfg(all(feature = "tiny-cli", any(not(debug_assertions), test)))]
mod tiny_main;
mod tmux_error_surface;

#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
use std::env;
#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
use std::ffi::OsString;
#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
use std::io::{self, ErrorKind, Write};
#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
use std::path::PathBuf;

#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
use rmux_client::INTERNAL_DAEMON_FLAG;
#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
use rmux_server::{ConfigFileSelection as ServerConfigFileSelection, DaemonConfig, ServerDaemon};

#[cfg(all(feature = "tiny-cli", not(debug_assertions)))]
fn main() {
    tiny_main::main();
}

#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
fn main() {
    match process_locale::initialize_process_locale()
        .map_err(|error| cli::ExitFailure::new(1, error))
        .and_then(|()| try_main(env::args_os()))
    {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            if !error.message().is_empty() {
                let _ = write_exit_message(error.message(), error.use_stderr());
            }
            std::process::exit(error.exit_code());
        }
    }
}

#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
fn write_exit_message(message: &str, stderr: bool) -> io::Result<()> {
    if stderr {
        match writeln!(io::stderr().lock(), "{message}") {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == ErrorKind::BrokenPipe => Ok(()),
            Err(error) => Err(error),
        }
    } else {
        match writeln!(io::stdout().lock(), "{message}") {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == ErrorKind::BrokenPipe => Ok(()),
            Err(error) => Err(error),
        }
    }
}

#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
fn try_main<I>(args: I) -> Result<i32, cli::ExitFailure>
where
    I: IntoIterator<Item = OsString>,
{
    let args: Vec<OsString> = args.into_iter().collect();

    match args.get(1) {
        Some(argument) if argument == INTERNAL_DAEMON_FLAG => {
            let internal = parse_internal_daemon_args(args.into_iter().skip(2))
                .map_err(|error| cli::ExitFailure::new(1, error))?;
            run_hidden_daemon(internal)
                .map_err(|error| error.to_string())
                .map(|()| 0)
                .map_err(|error| cli::ExitFailure::new(1, error))
        }
        _ => cli::run(args),
    }
}

#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct InternalDaemonArgs {
    socket_path: Option<PathBuf>,
    config_selection: ServerConfigFileSelection,
    config_quiet: bool,
    config_cwd: Option<PathBuf>,
    web_frontend: Option<String>,
    web_port: Option<u16>,
    startup_ready_fd: Option<i32>,
    startup_ready_event: Option<OsString>,
}

#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
#[cfg(test)]
fn parse_internal_socket_path<I>(args: I) -> Result<Option<PathBuf>, String>
where
    I: Iterator<Item = OsString>,
{
    parse_internal_daemon_args(args).map(|args| args.socket_path)
}

#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
fn parse_internal_daemon_args<I>(mut args: I) -> Result<InternalDaemonArgs, String>
where
    I: Iterator<Item = OsString>,
{
    let mut socket_path = None;
    let mut config_selection = ServerConfigFileSelection::Disabled;
    let mut config_quiet = false;
    let mut config_cwd = None;
    let mut web_frontend = None;
    let mut web_port = None;
    let mut startup_ready_fd = None;
    let mut startup_ready_event = None;

    if let Some(first) = args.next() {
        if os_string::os_str_bytes(first.as_os_str()).starts_with(b"--") {
            parse_internal_flag(
                first,
                &mut args,
                &mut config_selection,
                &mut config_quiet,
                &mut config_cwd,
                &mut web_frontend,
                &mut web_port,
                &mut startup_ready_fd,
                &mut startup_ready_event,
            )?;
        } else {
            socket_path = Some(PathBuf::from(first));
        }
    }

    while let Some(argument) = args.next() {
        if !os_string::os_str_bytes(argument.as_os_str()).starts_with(b"--") {
            return Err("unexpected extra arguments for hidden daemon mode".to_owned());
        }
        parse_internal_flag(
            argument,
            &mut args,
            &mut config_selection,
            &mut config_quiet,
            &mut config_cwd,
            &mut web_frontend,
            &mut web_port,
            &mut startup_ready_fd,
            &mut startup_ready_event,
        )?;
    }

    Ok(InternalDaemonArgs {
        socket_path,
        config_selection,
        config_quiet,
        config_cwd,
        web_frontend,
        web_port,
        startup_ready_fd,
        startup_ready_event,
    })
}

#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
#[allow(clippy::too_many_arguments)]
fn parse_internal_flag<I>(
    argument: OsString,
    args: &mut I,
    config_selection: &mut ServerConfigFileSelection,
    config_quiet: &mut bool,
    config_cwd: &mut Option<PathBuf>,
    web_frontend: &mut Option<String>,
    web_port: &mut Option<u16>,
    startup_ready_fd: &mut Option<i32>,
    startup_ready_event: &mut Option<OsString>,
) -> Result<(), String>
where
    I: Iterator<Item = OsString>,
{
    match argument.to_str() {
        Some("--config-default") => {
            if !matches!(config_selection, ServerConfigFileSelection::Disabled) {
                return Err("duplicate hidden daemon config selection".to_owned());
            }
            *config_selection = ServerConfigFileSelection::Default;
        }
        Some("--config-file") => {
            let file = args
                .next()
                .ok_or_else(|| "--config-file requires a path".to_owned())?;
            match config_selection {
                ServerConfigFileSelection::Disabled => {
                    *config_selection = ServerConfigFileSelection::Files(vec![PathBuf::from(file)]);
                }
                ServerConfigFileSelection::Files(files) => files.push(PathBuf::from(file)),
                ServerConfigFileSelection::Default => {
                    return Err("--config-file conflicts with --config-default".to_owned());
                }
            }
        }
        Some("--config-quiet") => *config_quiet = true,
        Some("--config-cwd") => {
            let cwd = args
                .next()
                .ok_or_else(|| "--config-cwd requires a path".to_owned())?;
            *config_cwd = Some(PathBuf::from(cwd));
        }
        Some("--web-port") => {
            let port = args
                .next()
                .ok_or_else(|| "--web-port requires a port".to_owned())?;
            let port = port
                .to_str()
                .ok_or_else(|| "invalid UTF-8 in --web-port".to_owned())?
                .parse::<u16>()
                .map_err(|_| "--web-port requires an integer port".to_owned())?;
            if port == 0 {
                return Err("--web-port must be between 1 and 65535".to_owned());
            }
            *web_port = Some(port);
        }
        Some("--frontend-url" | "--web-frontend") => {
            let frontend = args
                .next()
                .ok_or_else(|| "--frontend-url requires a URL".to_owned())?;
            let frontend = frontend
                .to_str()
                .ok_or_else(|| "invalid UTF-8 in --frontend-url".to_owned())?;
            *web_frontend = Some(frontend.to_owned());
        }
        Some("--startup-ready-fd") => {
            let fd = args
                .next()
                .ok_or_else(|| "--startup-ready-fd requires a file descriptor".to_owned())?;
            let fd = fd
                .to_str()
                .ok_or_else(|| "invalid UTF-8 in --startup-ready-fd".to_owned())?
                .parse::<i32>()
                .map_err(|_| "--startup-ready-fd requires an integer file descriptor".to_owned())?;
            if fd < 0 {
                return Err("--startup-ready-fd requires a non-negative file descriptor".to_owned());
            }
            *startup_ready_fd = Some(fd);
        }
        Some("--startup-ready-event") => {
            let event = args
                .next()
                .ok_or_else(|| "--startup-ready-event requires an event name".to_owned())?;
            *startup_ready_event = Some(event);
        }
        Some(other) => {
            return Err(format!("unexpected hidden daemon argument '{other}'"));
        }
        None => return Err("invalid UTF-8 in hidden daemon flag".to_owned()),
    }

    Ok(())
}

#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
fn run_hidden_daemon(args: InternalDaemonArgs) -> io::Result<()> {
    reject_unsupported_web_args(&args)?;

    let mut config = match args.socket_path {
        Some(socket_path) => DaemonConfig::new(socket_path),
        None => DaemonConfig::with_default_socket_path()?,
    };
    config = match args.config_selection {
        ServerConfigFileSelection::Disabled => config,
        ServerConfigFileSelection::Default => {
            config.with_default_config_load(args.config_quiet, args.config_cwd)
        }
        ServerConfigFileSelection::Files(files) => {
            config.with_config_files(files, args.config_quiet, args.config_cwd)
        }
    };
    if let Some(port) = args.web_port {
        config = config.with_web_port(port);
    }
    if let Some(frontend) = args.web_frontend {
        config = config.with_web_frontend(frontend);
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    if let Some(ready_fd) = args.startup_ready_fd {
        config = config.with_startup_ready_fd(ready_fd);
    }
    #[cfg(windows)]
    if let Some(ready_event) = args.startup_ready_event {
        config = config.with_startup_ready_event(ready_event);
    }
    rmux_os::memory::configure_daemon_allocator();
    let runtime = server_runtime::build_daemon_runtime()?;

    runtime.block_on(async move {
        let server = ServerDaemon::new(config).bind().await?;
        server.wait().await
    })
}

#[cfg(any(not(feature = "tiny-cli"), debug_assertions))]
fn reject_unsupported_web_args(args: &InternalDaemonArgs) -> io::Result<()> {
    #[cfg(not(feature = "web"))]
    if args.web_port.is_some() || args.web_frontend.is_some() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "rmux was built without web-share support",
        ));
    }

    #[cfg(feature = "web")]
    {
        let _ = args;
    }

    Ok(())
}

#[cfg(all(test, any(not(feature = "tiny-cli"), debug_assertions)))]
mod tests {
    use super::{parse_internal_daemon_args, parse_internal_socket_path, try_main};
    use rmux_client::INTERNAL_DAEMON_FLAG;
    use rmux_server::ConfigFileSelection;
    use std::ffi::OsString;
    use std::path::PathBuf;

    /// KOPITIAM ships `kmux`, not upstream's `rmux`.
    ///
    /// This guard is upstream's, and it earns its keep in the fork: the client
    /// finds its daemon by looking for a sibling binary *by file name* (see
    /// `rmux-client`'s `auto_start`), so the package name, the `[[bin]]` name
    /// and the name the daemon-discovery code searches for must agree. If they
    /// drift, `kmux` silently fails to auto-start its daemon at runtime rather
    /// than failing to build.
    const EXPECTED_BINARY_NAME: &str = "kmux";

    #[test]
    fn compiled_binary_name_is_kmux() {
        let compiled_binary_name = option_env!("CARGO_BIN_NAME").unwrap_or(env!("CARGO_PKG_NAME"));
        assert_eq!(compiled_binary_name, EXPECTED_BINARY_NAME);
    }

    #[test]
    fn hidden_daemon_parser_accepts_an_optional_socket_path() {
        let socket_path =
            parse_internal_socket_path([OsString::from("/tmp/rmux-hidden.sock")].into_iter())
                .expect("hidden socket path");

        assert_eq!(socket_path, Some(PathBuf::from("/tmp/rmux-hidden.sock")));
    }

    #[test]
    fn hidden_daemon_parser_rejects_unexpected_arguments() {
        let error = parse_internal_socket_path(
            [
                OsString::from("/tmp/rmux-hidden.sock"),
                OsString::from("/tmp/extra.sock"),
            ]
            .into_iter(),
        )
        .expect_err("unexpected hidden daemon argument should fail");

        assert!(error.contains("unexpected extra arguments"));
    }

    #[test]
    fn hidden_daemon_parser_defaults_to_the_spec_socket_when_unset() {
        let socket_path =
            parse_internal_socket_path(std::iter::empty()).expect("default socket path selection");

        assert_eq!(socket_path, None);
    }

    #[test]
    fn hidden_daemon_parser_accepts_config_forwarding_flags() {
        let args = parse_internal_daemon_args(
            [
                OsString::from("/tmp/rmux-hidden.sock"),
                OsString::from("--config-file"),
                OsString::from("one.conf"),
                OsString::from("--config-file"),
                OsString::from("two.conf"),
                OsString::from("--config-quiet"),
                OsString::from("--config-cwd"),
                OsString::from("/tmp/cwd"),
            ]
            .into_iter(),
        )
        .expect("hidden config args");

        assert_eq!(
            args.socket_path,
            Some(PathBuf::from("/tmp/rmux-hidden.sock"))
        );
        assert!(args.config_quiet);
        assert_eq!(args.config_cwd, Some(PathBuf::from("/tmp/cwd")));
        assert_eq!(
            args.config_selection,
            ConfigFileSelection::Files(vec![PathBuf::from("one.conf"), PathBuf::from("two.conf")])
        );
    }

    #[test]
    fn try_main_reports_absent_server_before_command_parse_failures() {
        #[cfg(unix)]
        let socket_args = [
            OsString::from("-S"),
            OsString::from(format!(
                "/tmp/rmux-main-missing-{}-parse.sock",
                std::process::id()
            )),
        ];
        #[cfg(windows)]
        let socket_args = [
            OsString::from("-L"),
            OsString::from(format!("main-missing-{}-parse", std::process::id())),
        ];

        let result = try_main([
            OsString::from("rmux"),
            socket_args[0].clone(),
            socket_args[1].clone(),
            OsString::from("new-session"),
            OsString::from("-s"),
        ]);

        let error = result.expect_err("missing new-session value should fail");
        assert_eq!(error.exit_code(), 1);
        assert!(
            error.message().contains("error connecting to"),
            "{}",
            error.message()
        );
    }

    #[test]
    fn try_main_rejects_hidden_daemon_extra_arguments() {
        let error = try_main([
            OsString::from("rmux"),
            OsString::from(INTERNAL_DAEMON_FLAG),
            OsString::from("/tmp/rmux-hidden.sock"),
            OsString::from("/tmp/extra.sock"),
        ])
        .expect_err("unexpected hidden daemon arguments should fail");

        assert!(error.message().contains("unexpected extra arguments"));
    }
}

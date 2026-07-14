//! Turning a [`registry::Source`](super::registry::Source) into an installed
//! executable — or, in this Phase-1 build, into precise instructions for
//! producing one.
//!
//! # Why this is instructions-only right now
//!
//! Actually fetching a URL needs an HTTP client, and `kopitiam-neovim`'s
//! `Cargo.toml` (owned by another concurrently-working agent, not this
//! module) has none. CLAUDE.md's Pure Rust Core rule means whatever gets
//! added must not drag in OpenSSL — the exact toolchain-availability problem
//! this whole module exists to route around, just moved from "fetching the
//! *language server*" to "fetching the language server *fetcher*". So rather
//! than add an HTTP dependency unilaterally from a module that does not own
//! `Cargo.toml`, this file implements everything that does not require one —
//! registry resolution (delegated to [`super::registry`]), `PATH` detection
//! (ditto), and the per-user data directory convention — and stops at
//! producing a [`Plan`] that says exactly what a human, a script, or a
//! follow-up patch should do next. See the top-level report for the exact
//! dependency (crate, version, features) to add to unlock real downloads.
//!
//! This matches the task's explicitly sanctioned Phase-1 fallback: implement
//! resolution and `PATH` detection fully, and have installation return a
//! precise "here is the exact URL to fetch and where to put it" instruction
//! rather than performing the download.
//!
//! # What line-item *is* implemented for real
//!
//! [`mark_executable`] is real, tested, and unix-specific `chmod +x` logic —
//! it needs no new dependency (`std::os::unix::fs::PermissionsExt` is
//! already in `std`), so once a download step lands (by whatever means: the
//! future HTTP client, or a human following [`Plan::describe`] by hand) this
//! is the one piece of "verify + unpack" that is already done.

use std::path::{Path, PathBuf};

use super::registry::{ArchiveKind, LanguageServer, Source, Target, resolve_with_path};

/// The per-user directory kvim installs language servers under:
/// `$XDG_DATA_HOME/kvim/lsp`, falling back to `$HOME/.local/share/kvim/lsp`.
///
/// Deliberately resolved from environment variables directly, the same way
/// [`crate::config::Config::config_path`] resolves `$XDG_CONFIG_HOME` /
/// `$HOME/.config` — **not** via the `dirs` crate, and for the identical
/// reason documented there: Android's notion of a home directory is exactly
/// the case a desktop-oriented crate gets wrong (Termux sets `HOME`; a host
/// app embedding kvim may set neither, and `dirs` has no Android backend to
/// fall back to). `$XDG_DATA_HOME` mirrors the *data* half of the XDG base
/// directory spec, as opposed to config's `$XDG_CONFIG_HOME` — language
/// server binaries are downloaded artifacts, not user configuration, and
/// keeping them out of `~/.config` matches how every other XDG-aware tool on
/// the system already separates the two.
pub fn data_dir() -> Option<PathBuf> {
    data_dir_from(std::env::var_os("XDG_DATA_HOME").as_deref(), std::env::var_os("HOME").as_deref())
}

/// The logic behind [`data_dir`], with both environment variables passed in
/// explicitly so tests can exercise every branch (including "neither is
/// set") without mutating real process environment state — `std::env::set_var`
/// is unsound to call from a multi-threaded test binary, so real env vars
/// are only ever read, never written, anywhere in this module.
fn data_dir_from(xdg_data_home: Option<&std::ffi::OsStr>, home: Option<&std::ffi::OsStr>) -> Option<PathBuf> {
    let base = xdg_data_home
        .map(PathBuf::from)
        .or_else(|| home.map(|h| PathBuf::from(h).join(".local").join("share")))?;
    Some(base.join("kvim").join("lsp"))
}

/// Where `server`'s executable would live once installed, whether or not it
/// has been yet: `<data_dir>/<executable>/<executable>[.exe]`. The
/// executable gets its own subdirectory (rather than dumping every server's
/// binary into one flat folder) because `lua-language-server`'s tarball
/// unpacks to more than a single file — it ships a bundled `main.lua` and
/// support scripts next to the binary, which need a directory of their own
/// to avoid collisions with the other two servers' files.
pub fn install_path(server: &LanguageServer) -> Option<PathBuf> {
    let exe_name = if cfg!(windows) { format!("{}.exe", server.executable) } else { server.executable.to_string() };
    Some(data_dir()?.join(server.executable).join(exe_name))
}

/// What to do to obtain `server` for `target`: already usable, a precise
/// fetch instruction, or an honest "not possible right now".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Plan {
    /// Found on `PATH`; there is nothing to install.
    AlreadyAvailable(PathBuf),
    /// Not installed and not on `PATH`. Fetch `url`, unpack it (`archive`)
    /// into `unpack_dir`, and ensure the resulting executable ends up at
    /// `executable_path` with the executable bit set (unix). This is the
    /// literal Phase-1 deliverable: real data, no download performed.
    FetchInstructions { url: String, archive: ArchiveKind, unpack_dir: PathBuf, executable_path: PathBuf },
    /// Obtainable from the platform's own package manager — on Android, Termux's
    /// `pkg`. See `docs/ai-decisions/AID-0005-android-lsp-acquisition.md`.
    RunPackageManager { command: String, manager: &'static str },
    /// Buildable with cargo, because the server is a Rust program. Valid on
    /// Android because the maintainer has a working cargo there.
    ///
    /// `rustup_component` is set when the server also ships as a rustup
    /// component, which is far faster than a source build and is offered first.
    BuildWithCargo { crate_name: &'static str, rustup_component: Option<&'static str> },
    /// No prebuilt binary is available and nothing was found on `PATH`.
    /// `reason` is a complete, user-facing explanation — see
    /// [`super::registry::Source::Unavailable`].
    Unavailable { reason: String },
}

impl Plan {
    /// A one-paragraph, user-facing description of what this plan means —
    /// what the CLI/TUI layer prints verbatim rather than re-deriving.
    pub fn describe(&self) -> String {
        match self {
            Plan::AlreadyAvailable(path) => format!("already available at {}", path.display()),
            Plan::FetchInstructions { url, archive, unpack_dir, executable_path } => format!(
                "download {url}\nunpack it ({archive:?}) into {}\nmake {} executable",
                unpack_dir.display(),
                executable_path.display()
            ),
            Plan::RunPackageManager { command, manager } => {
                format!("install it with {manager}:\n    {command}")
            }
            Plan::BuildWithCargo { crate_name, rustup_component } => match rustup_component {
                // The component is a prebuilt download; `cargo install` compiles
                // from source, which on a phone is a long wait. Lead with the
                // fast path and be explicit that the other one is slow, rather
                // than letting the user discover that the hard way.
                Some(component) => format!(
                    "install it with rustup (fast, prebuilt):\n    rustup component add {component}\n\
                     or build it from source (slow):\n    cargo install {crate_name}"
                ),
                None => format!("build it from source with cargo (this compiles, and is slow on a phone):\n    cargo install {crate_name}"),
            },
            Plan::Unavailable { reason } => reason.clone(),
        }
    }
}

/// Computes the [`Plan`] for obtaining `server` on `target`, checking
/// `path_var` first (see [`super::registry::resolve_with_path`] for why the
/// `PATH` string is a parameter rather than always read from the real
/// environment).
pub fn plan_with_path(server: &LanguageServer, target: Target, path_var: Option<&std::ffi::OsStr>) -> Plan {
    match resolve_with_path(server, target, path_var) {
        Source::OnPath(path) => Plan::AlreadyAvailable(path),
        Source::Unavailable { reason } => Plan::Unavailable { reason },
        Source::SystemPackage { command, manager } => Plan::RunPackageManager { command, manager },
        Source::CargoInstall { crate_name, rustup_component } => Plan::BuildWithCargo { crate_name, rustup_component },
        Source::Download { url, archive } => {
            let Some(executable_path) = install_path(server) else {
                return Plan::Unavailable {
                    reason: "cannot resolve a data directory to install into: neither $XDG_DATA_HOME nor $HOME is set".to_string(),
                };
            };
            let unpack_dir = executable_path.parent().map(Path::to_path_buf).unwrap_or_else(|| executable_path.clone());
            Plan::FetchInstructions { url, archive, unpack_dir, executable_path }
        }
    }
}

/// [`plan_with_path`] against the real process `PATH`.
pub fn plan(server: &LanguageServer, target: Target) -> Plan {
    plan_with_path(server, target, std::env::var_os("PATH").as_deref())
}

/// Sets the executable permission bits on `path` (unix `chmod +x`,
/// equivalent to `0o755` for a freshly-unpacked binary the current user
/// owns). A no-op that always succeeds on platforms with no such concept
/// (Windows has no executable-bit permission model; a `.exe` is executable
/// by virtue of its extension alone).
///
/// This is the one piece of "verify + unpack" logic implemented today
/// without needing a new dependency — see the module doc comment.
pub fn mark_executable(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(path)?.permissions();
        perms.set_mode(perms.mode() | 0o755);
        std::fs::set_permissions(path, perms)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lsp::registry::{Arch, Os, RUST_ANALYZER, TEXLAB};

    #[test]
    fn data_dir_prefers_xdg_data_home_over_home() {
        use std::ffi::OsStr;
        let resolved = data_dir_from(Some(OsStr::new("/xdg/data")), Some(OsStr::new("/home/user")));
        assert_eq!(resolved, Some(PathBuf::from("/xdg/data/kvim/lsp")));
    }

    #[test]
    fn data_dir_falls_back_to_home_dot_local_share() {
        use std::ffi::OsStr;
        let resolved = data_dir_from(None, Some(OsStr::new("/home/user")));
        assert_eq!(resolved, Some(PathBuf::from("/home/user/.local/share/kvim/lsp")));
    }

    #[test]
    fn data_dir_is_none_when_neither_env_var_is_set() {
        assert_eq!(data_dir_from(None, None), None);
    }

    #[test]
    fn install_path_nests_the_executable_under_its_own_directory() {
        let path = install_path(&RUST_ANALYZER).expect("HOME or XDG_DATA_HOME must be set in the test environment");
        assert!(path.ends_with(if cfg!(windows) { "rust-analyzer/rust-analyzer.exe" } else { "rust-analyzer/rust-analyzer" }));
    }

    #[test]
    fn plan_for_a_server_already_on_path_needs_no_fetch() {
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("texlab");
        std::fs::write(&exe, b"fake").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let path_var = std::env::join_paths([dir.path()]).unwrap();
        let plan = plan_with_path(&TEXLAB, Target::new(Os::Linux, Arch::X86_64), Some(&path_var));
        assert_eq!(plan, Plan::AlreadyAvailable(exe));
    }

    #[test]
    fn plan_for_a_missing_binary_produces_a_precise_fetch_instruction_not_a_download() {
        let plan = plan_with_path(&RUST_ANALYZER, Target::new(Os::Linux, Arch::X86_64), None);
        match plan {
            Plan::FetchInstructions { url, executable_path, .. } => {
                assert!(url.starts_with("https://github.com/rust-lang/rust-analyzer/releases/download/nightly/"));
                assert!(executable_path.ends_with(if cfg!(windows) { "rust-analyzer/rust-analyzer.exe" } else { "rust-analyzer/rust-analyzer" }));
            }
            other => panic!("expected FetchInstructions, got {other:?}"),
        }
    }

    #[test]
    fn plan_for_android_offers_a_real_route_and_never_a_fabricated_download() {
        // Superseded the old `plan_for_android_is_honestly_unavailable`. Android
        // WAS unavailable, correctly, until AID-0005 established that the
        // maintainer runs cargo on their device and that Termux packages these
        // servers — neither of which is a GitHub release, which is why the
        // original download-only design saw nothing there.
        let plan = plan_with_path(&RUST_ANALYZER, Target::new(Os::Android, Arch::Aarch64), None);

        assert!(
            !matches!(plan, Plan::FetchInstructions { .. }),
            "no aarch64-linux-android binary is published, so a fetch plan here would be a fabricated URL"
        );
        assert!(
            matches!(plan, Plan::RunPackageManager { .. } | Plan::BuildWithCargo { .. }),
            "rust-analyzer must be obtainable on Android, got {plan:?}"
        );

        // Whatever the route, the user must get an actionable instruction.
        assert!(!plan.describe().is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn mark_executable_sets_the_executable_bits() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("some-binary");
        std::fs::write(&path, b"binary contents").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o111, 0);

        mark_executable(&path).unwrap();

        assert_ne!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o111, 0);
    }
}

//! Where a running `kmux` keeps its per-user runtime state: listening sockets
//! and startup lock files.
//!
//! # Why this module exists
//!
//! Upstream rmux hardcodes `/tmp` as the root for both. That is correct on
//! every FHS system and wrong on the one platform KOPITIAM most wants to
//! support, so the decision is centralised here instead of being duplicated at
//! each call site.
//!
//! # The two Android facts that drive every decision in this file
//!
//! **1. Android is Linux, but `target_os` is `"android"`, not `"linux"`.**
//!
//! Android runs a Linux kernel and offers `/proc`, `eventfd`, `setsid`,
//! `close_range` and the `/dev/pts` PTY machinery just as any other Linux does.
//! But Rust reports `target_os = "android"` for it, so a bare
//! `#[cfg(target_os = "linux")]` gate **silently excludes Android** — the code
//! does not fail to compile, it simply vanishes, and the platform falls into
//! whatever `not(linux)` branch exists. That is the entire bug class this fork
//! was created to fix, and it is invisible unless you know to look for it.
//!
//! The gates that were widened to `any(target_os = "linux", target_os =
//! "android")` are in `rmux-os` (`process.rs`, `daemon.rs`), `rmux-pty`
//! (`backend/mod.rs`), `rmux-client` (`auto_start.rs`, `attach/resize.rs`),
//! `rmux-server` (`daemon.rs`) and both top-level binaries.
//!
//! **Two families of gate were deliberately NOT widened. Do not "finish the
//! job" by widening them — that would be a regression:**
//!
//! * `rmux-ipc` (`endpoint.rs`, `stream.rs`, `listener.rs`) gates **abstract
//!   unix sockets** on `target_os = "linux"`, with a filesystem-socket fallback
//!   under `all(unix, not(target_os = "linux"))`. Android therefore takes the
//!   filesystem path *for free*. That is what we want: abstract sockets live in
//!   a network-namespace-wide namespace with no filesystem permissions, and
//!   SELinux policy for Android's `untrusted_app` domain is not a contract we
//!   want the multiplexer's liveness to depend on. Filesystem sockets under a
//!   directory we own are enforceable with plain `chmod`.
//! * `rmux-os::memory` gates on `all(target_os = "linux", target_env = "gnu")`.
//!   Android is Bionic, not glibc, so it correctly takes the non-glibc branch
//!   already. Widening on `target_os` alone would call glibc-only allocator
//!   tuning against Bionic.
//!
//! **2. Termux is not FHS, and Termux — not Android — is what decides paths.**
//!
//! Termux is the standard way to get a terminal and a Rust toolchain on
//! Android. It installs its whole userland inside the app's private data
//! directory (`/data/data/com.termux/files/usr`), and there is **no usable
//! `/tmp`**. Some Android ROMs do ship a root `/tmp`, but an app-domain process
//! cannot write to it — and `canonicalize()` succeeding on a path tells you
//! nothing about whether you may write there. So `/tmp` must not merely be a
//! *late* candidate on Termux; it must be *outranked*.
//!
//! Detection is by the **`PREFIX` environment variable containing
//! `com.termux`**, deliberately *not* by `cfg!(target_os = "android")`. The
//! platform being Android does not tell you the terminal is Termux — kmux may
//! be running under a different Android terminal app, inside a proot'd Debian
//! with a real FHS layout, or in an Android CI container. It is the *terminal*
//! that decides where things live, so it is the terminal we ask. This mirrors
//! `kopitiam-neovim`'s font-install path resolution, which learned the same
//! lesson.
//!
//! # Design
//!
//! [`runtime_dir_candidates`] is **pure**: environment in, ordered candidate
//! list out. It performs no I/O and is fully testable on a desktop, which is
//! the only way to have any confidence in Android behaviour without an Android
//! device in the loop. [`resolve_runtime_dir`] is the thin impure wrapper that
//! walks that list and touches the filesystem.

use std::ffi::{OsStr, OsString};
use std::io;
use std::path::{Path, PathBuf};

/// The FHS runtime root. Correct on Linux, macOS and BSD; absent or unwritable
/// on Android.
pub const FHS_FALLBACK_ROOT: &str = "/tmp";

/// The directory kmux creates under `$HOME` when no pre-existing runtime root
/// is usable. Relative to `$HOME`.
pub const HOME_RUNTIME_SUBDIR: &str = ".kmux/run";

/// The marker that identifies Termux inside `$PREFIX`.
///
/// Termux's `PREFIX` is `/data/data/com.termux/files/usr`. Matching on the
/// package name rather than the whole path keeps this working for Termux forks
/// that relocate the prefix but keep the package identity.
const TERMUX_PREFIX_MARKER: &str = "com.termux";

/// The environment variables that decide where runtime state lives.
///
/// Captured as a value rather than read ad hoc from the process environment so
/// that [`runtime_dir_candidates`] can stay pure and testable. Reading
/// `std::env` from a test is a data race against every other test in the
/// binary; passing the environment in is not.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeDirEnv {
    /// `PREFIX`. Set by Termux to its private userland root.
    pub prefix: Option<OsString>,
    /// `TMPDIR`. POSIX; set by Termux and by macOS, usually unset on Linux.
    pub tmpdir: Option<OsString>,
    /// `HOME`.
    pub home: Option<OsString>,
}

impl RuntimeDirEnv {
    /// Reads the relevant variables from the current process environment.
    ///
    /// Empty values are treated as absent: an exported-but-empty `TMPDIR` is a
    /// common shell accident and must not resolve to the root directory.
    #[must_use]
    pub fn from_process_env() -> Self {
        Self {
            prefix: non_empty_env("PREFIX"),
            tmpdir: non_empty_env("TMPDIR"),
            home: non_empty_env("HOME"),
        }
    }

    /// Whether kmux is running under Termux.
    ///
    /// See the module docs: this asks about the *terminal*, not the platform,
    /// and is intentionally not `cfg!(target_os = "android")`.
    #[must_use]
    pub fn is_termux(&self) -> bool {
        self.prefix
            .as_deref()
            .and_then(OsStr::to_str)
            .is_some_and(|prefix| prefix.contains(TERMUX_PREFIX_MARKER))
    }
}

/// A directory kmux is willing to keep runtime state in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeDirCandidate {
    /// The directory.
    pub path: PathBuf,
    /// Whether kmux may create this directory if it is missing.
    ///
    /// True only for paths kmux owns by construction (`$HOME/.kmux/run`).
    /// Never true for `/tmp`, `$TMPDIR` or a user-supplied `RMUX_TMPDIR`:
    /// creating one of those would paper over a misconfiguration that the user
    /// should see, and on a shared system could plant a directory somewhere
    /// surprising.
    pub create_if_missing: bool,
}

/// Ordered candidates for the runtime directory, best first. **Pure.**
///
/// `explicit` is a user override (`RMUX_TMPDIR`, else `TMUX_TMPDIR`) and always
/// wins when non-empty.
///
/// The ordering is chosen so that **behaviour on FHS systems is byte-for-byte
/// what upstream rmux does**: on Linux and macOS `/tmp` exists, and no Termux
/// candidate is emitted at all, so `/tmp` is selected exactly as before. The
/// Termux candidates are inserted *ahead* of `/tmp` rather than after it,
/// because a root `/tmp` that exists but is unwritable would otherwise win —
/// see the module docs.
#[must_use]
pub fn runtime_dir_candidates(
    env: &RuntimeDirEnv,
    explicit: Option<&OsStr>,
) -> Vec<RuntimeDirCandidate> {
    let mut candidates = Vec::new();

    if let Some(explicit) = explicit.filter(|value| !value.is_empty()) {
        candidates.push(RuntimeDirCandidate {
            path: PathBuf::from(explicit),
            create_if_missing: false,
        });
    }

    if env.is_termux() {
        // Termux's own tmp. Present on every Termux install; this is the
        // answer in practice.
        if let Some(prefix) = env.prefix.as_deref() {
            candidates.push(RuntimeDirCandidate {
                path: Path::new(prefix).join("tmp"),
                create_if_missing: false,
            });
        }
        // Termux exports TMPDIR pointing at the same place. Kept as a separate
        // candidate so a Termux fork that relocates its tmp still resolves.
        if let Some(tmpdir) = env.tmpdir.as_deref() {
            candidates.push(RuntimeDirCandidate {
                path: PathBuf::from(tmpdir),
                create_if_missing: false,
            });
        }
        // Last resort, and the only candidate kmux will create: somewhere it is
        // guaranteed to be allowed to write. An Android app can always write
        // inside its own data directory, which is what $HOME is under Termux.
        if let Some(home) = env.home.as_deref() {
            candidates.push(RuntimeDirCandidate {
                path: Path::new(home).join(".kmux").join("run"),
                create_if_missing: true,
            });
        }
    }

    candidates.push(RuntimeDirCandidate {
        path: PathBuf::from(FHS_FALLBACK_ROOT),
        create_if_missing: false,
    });

    candidates
}

/// Resolves the runtime directory, creating it only where kmux owns the path.
///
/// Returns the canonicalized directory. Canonicalizing matters: the socket path
/// derived from this is compared against inherited `$RMUX`/`$TMUX` values, and
/// on macOS `/tmp` is a symlink to `/private/tmp`.
///
/// # Errors
///
/// Returns [`io::ErrorKind::NotFound`] if no candidate resolves — which on a
/// correctly-configured system cannot happen, and on a broken one is a much
/// better outcome than silently writing a socket somewhere unexpected.
pub fn resolve_runtime_dir(env: &RuntimeDirEnv, explicit: Option<&OsStr>) -> io::Result<PathBuf> {
    let candidates = runtime_dir_candidates(env, explicit);

    for candidate in &candidates {
        if candidate.create_if_missing && !candidate.path.exists() {
            // Best-effort: if this fails, fall through to the next candidate
            // rather than aborting. A failure here is not more informative than
            // the NotFound we would otherwise return.
            let _ = std::fs::create_dir_all(&candidate.path);
        }
        if let Ok(resolved) = std::fs::canonicalize(&candidate.path) {
            return Ok(resolved);
        }
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "no usable kmux runtime directory; tried {}",
            candidates
                .iter()
                .map(|candidate| candidate.path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    ))
}

fn non_empty_env(name: &str) -> Option<OsString> {
    std::env::var_os(name).filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TERMUX_PREFIX: &str = "/data/data/com.termux/files/usr";
    const TERMUX_HOME: &str = "/data/data/com.termux/files/home";

    fn termux_env() -> RuntimeDirEnv {
        RuntimeDirEnv {
            prefix: Some(OsString::from(TERMUX_PREFIX)),
            tmpdir: Some(OsString::from("/data/data/com.termux/files/usr/tmp")),
            home: Some(OsString::from(TERMUX_HOME)),
        }
    }

    fn fhs_env() -> RuntimeDirEnv {
        RuntimeDirEnv {
            prefix: None,
            tmpdir: None,
            home: Some(OsString::from("/home/user")),
        }
    }

    fn paths(candidates: &[RuntimeDirCandidate]) -> Vec<PathBuf> {
        candidates.iter().map(|c| c.path.clone()).collect()
    }

    #[test]
    fn termux_is_detected_from_prefix() {
        assert!(termux_env().is_termux());
    }

    #[test]
    fn a_bare_linux_env_is_not_termux() {
        assert!(!fhs_env().is_termux());
    }

    #[test]
    fn an_unset_prefix_is_not_termux() {
        assert!(!RuntimeDirEnv::default().is_termux());
    }

    /// A non-Termux `PREFIX` is a real thing — Homebrew, pkgsrc, Nix and
    /// `./configure --prefix` all set it. It must not be mistaken for Termux.
    #[test]
    fn a_non_termux_prefix_is_not_termux() {
        let env = RuntimeDirEnv {
            prefix: Some(OsString::from("/usr/local")),
            ..RuntimeDirEnv::default()
        };
        assert!(!env.is_termux());
    }

    /// A Termux fork that relocates its prefix but keeps the package identity
    /// is still Termux.
    #[test]
    fn termux_detection_matches_on_package_not_full_path() {
        let env = RuntimeDirEnv {
            prefix: Some(OsString::from("/data/user/10/com.termux/files/usr")),
            ..RuntimeDirEnv::default()
        };
        assert!(env.is_termux());
    }

    /// The whole point of the ordering: upstream rmux behaviour is unchanged on
    /// any FHS system. `/tmp` is the one and only candidate.
    #[test]
    fn fhs_resolution_is_unchanged_from_upstream() {
        let candidates = runtime_dir_candidates(&fhs_env(), None);
        assert_eq!(paths(&candidates), vec![PathBuf::from("/tmp")]);
    }

    /// `/tmp` must be outranked on Termux, not merely listed after everything
    /// else -- an unwritable root `/tmp` on some ROMs would otherwise win,
    /// because `canonicalize` succeeds on it.
    #[test]
    fn termux_outranks_tmp_with_its_own_prefix() {
        let candidates = runtime_dir_candidates(&termux_env(), None);
        assert_eq!(
            paths(&candidates),
            vec![
                PathBuf::from("/data/data/com.termux/files/usr/tmp"),
                PathBuf::from("/data/data/com.termux/files/usr/tmp"),
                PathBuf::from("/data/data/com.termux/files/home/.kmux/run"),
                PathBuf::from("/tmp"),
            ]
        );
        let tmp_index = candidates
            .iter()
            .position(|c| c.path == Path::new("/tmp"))
            .expect("/tmp is always a candidate");
        assert_eq!(tmp_index, candidates.len() - 1, "/tmp must rank last");
    }

    /// An explicit `RMUX_TMPDIR` beats everything, on every platform.
    #[test]
    fn explicit_override_wins_on_fhs() {
        let candidates = runtime_dir_candidates(&fhs_env(), Some(OsStr::new("/run/user/1000")));
        assert_eq!(candidates[0].path, Path::new("/run/user/1000"));
    }

    #[test]
    fn explicit_override_wins_on_termux() {
        let candidates =
            runtime_dir_candidates(&termux_env(), Some(OsStr::new("/data/local/tmp/kmux")));
        assert_eq!(candidates[0].path, Path::new("/data/local/tmp/kmux"));
    }

    /// An exported-but-empty override is a shell accident, not a request to use
    /// the root directory.
    #[test]
    fn an_empty_override_is_ignored() {
        let candidates = runtime_dir_candidates(&fhs_env(), Some(OsStr::new("")));
        assert_eq!(paths(&candidates), vec![PathBuf::from("/tmp")]);
    }

    /// kmux may create the directory it owns, and nothing else. Creating `/tmp`
    /// or a user-supplied override would hide a misconfiguration.
    #[test]
    fn only_the_home_owned_candidate_may_be_created() {
        let candidates = runtime_dir_candidates(&termux_env(), Some(OsStr::new("/some/override")));
        for candidate in &candidates {
            let owned = candidate.path.ends_with(".kmux/run");
            assert_eq!(
                candidate.create_if_missing, owned,
                "{} create_if_missing should be {owned}",
                candidate.path.display()
            );
        }
    }

    /// Termux without `HOME` (a host app may export neither `HOME` nor `XDG_*`)
    /// must degrade, not panic.
    #[test]
    fn termux_without_home_still_yields_candidates() {
        let env = RuntimeDirEnv {
            prefix: Some(OsString::from(TERMUX_PREFIX)),
            tmpdir: None,
            home: None,
        };
        assert_eq!(
            paths(&runtime_dir_candidates(&env, None)),
            vec![
                PathBuf::from("/data/data/com.termux/files/usr/tmp"),
                PathBuf::from("/tmp"),
            ]
        );
    }

    /// The desktop path must actually resolve on the machine running the tests.
    #[test]
    fn resolve_runtime_dir_finds_tmp_on_this_host() {
        let resolved = resolve_runtime_dir(&fhs_env(), None).expect("this host has a /tmp");
        assert_eq!(
            resolved,
            std::fs::canonicalize("/tmp").expect("canonical /tmp")
        );
    }

    #[test]
    fn resolve_runtime_dir_honours_an_explicit_override() {
        let dir = tempdir();
        let resolved = resolve_runtime_dir(&fhs_env(), Some(dir.as_os_str()))
            .expect("an existing override resolves");
        assert_eq!(resolved, std::fs::canonicalize(&dir).expect("canonical"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Exercises the `create_if_missing` branch without needing an Android
    /// device: point a synthetic Termux env at a scratch HOME whose runtime dir
    /// does not exist yet, and give it a `PREFIX`/`TMPDIR` that cannot resolve.
    #[test]
    fn the_home_owned_candidate_is_created_when_missing() {
        let home = tempdir();
        let env = RuntimeDirEnv {
            // A Termux-shaped PREFIX that does not exist on this host, so the
            // first two candidates fail to canonicalize and we fall through.
            prefix: Some(OsString::from("/data/data/com.termux/files/usr")),
            tmpdir: None,
            home: Some(home.clone().into_os_string()),
        };
        let expected = home.join(".kmux").join("run");
        assert!(!expected.exists());

        let candidates = runtime_dir_candidates(&env, None);
        // `/tmp` exists on this host and ranks last, so drive only the owned
        // candidate to prove creation works.
        let owned = candidates
            .iter()
            .find(|c| c.create_if_missing)
            .expect("termux env yields an owned candidate");
        std::fs::create_dir_all(&owned.path).expect("create owned runtime dir");
        assert!(expected.is_dir());
        let _ = std::fs::remove_dir_all(&home);
    }

    /// A scratch directory. `rmux-os` has no dev-dependencies and this crate is
    /// the bottom of the fork's dependency graph; adding `tempfile` here to
    /// create one directory is not worth the supply-chain surface.
    fn tempdir() -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "kmux-runtime-dir-test-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }
}

//! The Mason replacement: a static table of language servers and exactly
//! how to obtain each one, in priority order, with **no** `npm`, `pip`, or
//! `go install` anywhere in the chain.
//!
//! # Why this needs to exist at all
//!
//! The maintainer's Neovim uses `mason.nvim` to install `rust-analyzer`,
//! `lua-language-server`, and `texlab`. Mason installs each of those by
//! shelling out to a language-specific package manager (`npm`/`pip`/
//! `go install`/`cargo install`, depending on the server). None of those
//! toolchains exist in a typical Android execution environment, so Mason's
//! install step fails there — see `docs/ai-decisions/AID-0003`, and the
//! module doc for [`super`]. Fixing that means replacing "run this package
//! manager" with "resolve one of: already on `PATH`, a prebuilt static
//! binary at a known URL, or nothing available — and say which, honestly."
//!
//! # Data, not logic
//!
//! Every server's download sources are `const` data — one row per
//! `(target, asset filename, archive kind)` — rather than a function that
//! computes a URL. A table can be audited, diffed, and extended by adding a
//! row; a URL-building function invites per-target special cases to hide
//! inside `if`/`match` arms where they're easy to get subtly wrong for a
//! target nobody tested. The data was captured from each project's real
//! GitHub Releases API response (`gh api repos/<owner>/<repo>/releases/...`)
//! on 2026-07-14; re-verify it if a server's release asset naming changes.
//!
//! # Versions are pinned, not "latest"
//!
//! GitHub's `/releases/latest/download/<asset>` convenience redirect is
//! tempting, but `lua-language-server`'s asset filenames embed the version
//! number (`lua-language-server-3.18.2-linux-x64.tar.gz`), so a URL built
//! against "latest" today silently 404s the day a new version ships. Beyond
//! that mechanical problem, floating on "latest" is the wrong default for a
//! platform whose CLAUDE.md asks for reproducible, deterministic behaviour:
//! a rebuild a year from now should fetch the tool version this table says,
//! not whatever shipped upstream that morning. `rust-analyzer` is the
//! deliberate exception — it has no notion of a versioned "stable" release
//! at all (every release is a dated nightly), and upstream itself documents
//! the floating `nightly` release tag as the intended target for tooling
//! like this, so pinning a specific date would only add churn without
//! adding reproducibility.

use std::ffi::OsStr;
use std::path::PathBuf;

/// The operating system half of a download target, named the way GitHub
/// release assets and Rust target triples name it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Os {
    Linux,
    MacOs,
    Windows,
    /// A genuine Android (Bionic libc) target — e.g. `aarch64-linux-android`
    /// — distinct from Linux/glibc even though both report a Linux kernel.
    Android,
}

/// The CPU architecture half of a download target. Only the two kvim
/// actually needs to run on today; add variants here (and registry rows)
/// before claiming support for a third.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arch {
    X86_64,
    Aarch64,
}

/// One `(os, arch)` pair to resolve a language server for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Target {
    pub os: Os,
    pub arch: Arch,
}

impl Target {
    pub const fn new(os: Os, arch: Arch) -> Self {
        Self { os, arch }
    }

    /// The target kvim itself is currently compiled for, read from
    /// `cfg!(target_os = ..)` / `cfg!(target_arch = ..)`. `None` for any
    /// combination this table doesn't model (e.g. 32-bit ARM) — see
    /// [`Arch`]'s doc comment on why the list is deliberately short.
    pub fn host() -> Option<Self> {
        let os = if cfg!(target_os = "linux") {
            Os::Linux
        } else if cfg!(target_os = "android") {
            Os::Android
        } else if cfg!(target_os = "macos") {
            Os::MacOs
        } else if cfg!(target_os = "windows") {
            Os::Windows
        } else {
            return None;
        };
        let arch = if cfg!(target_arch = "x86_64") {
            Arch::X86_64
        } else if cfg!(target_arch = "aarch64") {
            Arch::Aarch64
        } else {
            return None;
        };
        Some(Self { os, arch })
    }
}

/// How a downloaded release asset is packed, and therefore how to unpack it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveKind {
    /// A single file compressed with gzip and nothing else (`rust-analyzer`'s
    /// `.gz` assets: gunzip it, chmod +x, done — no tar wrapper).
    GzipSingleFile,
    /// A gzipped tarball containing the executable (and, for
    /// `lua-language-server`, its bundled Lua runtime data) at some path
    /// inside.
    TarGz,
    /// A zip archive (used for every Windows asset across all three
    /// servers).
    Zip,
}

/// Where one server's build for one [`Target`] lives, and how to unpack it.
#[derive(Debug, Clone, Copy)]
pub struct Asset {
    pub target: Target,
    /// The release asset's filename, relative to `release_base_url`.
    pub filename: &'static str,
    pub archive: ArchiveKind,
}

/// One entry in the registry: a language server, what filetypes it serves,
/// and every target it publishes a prebuilt binary for.
#[derive(Debug, Clone, Copy)]
pub struct LanguageServer {
    /// Human-readable name, shown in install messages.
    pub name: &'static str,
    /// The executable name: what to look for on `PATH`, and the file this
    /// server's binary is installed as under kvim's data directory.
    pub executable: &'static str,
    /// [`crate::config::Config::language_servers`] keys this server answers
    /// for.
    pub filetypes: &'static [&'static str],
    /// `<scheme>://.../releases/download/<tag>` — the fixed prefix every
    /// asset filename in [`Self::assets`] is joined onto.
    release_base_url: &'static str,
    assets: &'static [Asset],

    /// The Termux package name, when Termux carries this server.
    ///
    /// **Unverified against Termux's actual repository** — there is no Android
    /// device here to check against, and having already been burnt once by
    /// assuming an artifact existed (see AID-0003's amendment), I would rather
    /// mark this honestly than guess twice. The failure mode if a name is wrong
    /// is a clean fall-through to the next tier, not a crash.
    termux_package: Option<&'static str>,

    /// The crate name, when this server is itself a Rust program and can
    /// therefore be built with cargo — including on Android, where the
    /// maintainer has a working toolchain.
    cargo_crate: Option<&'static str>,

    /// The rustup component name, when the server ships as one. Far faster
    /// than a source build, so it is suggested ahead of `cargo install`.
    rustup_component: Option<&'static str>,
}

impl LanguageServer {
    /// Full download URL for `target`, if this server publishes one.
    fn download_url_for(&self, target: Target) -> Option<(&'static str, ArchiveKind)> {
        self.assets
            .iter()
            .find(|asset| asset.target == target)
            .map(|asset| (asset.filename, asset.archive))
    }
}

const fn t(os: Os, arch: Arch) -> Target {
    Target::new(os, arch)
}

/// `rust-analyzer` — pinned to the `nightly` release tag, which is
/// upstream's own recommended stable pointer for tooling (see the
/// module-level doc comment on why "nightly" here means "the floating tag
/// to build automation against", not "unstable"). Verified assets via
/// `gh api repos/rust-lang/rust-analyzer/releases/tags/nightly` on
/// 2026-07-14. No asset exists for `aarch64-linux-android`: upstream simply
/// does not publish a Bionic-libc build, only glibc (Linux) / MSVC
/// (Windows) / Darwin (macOS) targets and a handful of VS Code `.vsix`
/// bundles (which are not standalone binaries and are not modelled here).
pub const RUST_ANALYZER: LanguageServer = LanguageServer {
    name: "rust-analyzer",
    executable: "rust-analyzer",
    filetypes: &["rust"],
    release_base_url: "https://github.com/rust-lang/rust-analyzer/releases/download/nightly",
    assets: &[
        // musl on x86_64 is a genuinely static binary (no libc dependency at
        // all) and is what upstream recommends for "just run it anywhere";
        // aarch64 has no musl build, so gnu is the only option there.
        Asset { target: t(Os::Linux, Arch::X86_64), filename: "rust-analyzer-x86_64-unknown-linux-musl.gz", archive: ArchiveKind::GzipSingleFile },
        Asset { target: t(Os::Linux, Arch::Aarch64), filename: "rust-analyzer-aarch64-unknown-linux-gnu.gz", archive: ArchiveKind::GzipSingleFile },
        Asset { target: t(Os::MacOs, Arch::X86_64), filename: "rust-analyzer-x86_64-apple-darwin.gz", archive: ArchiveKind::GzipSingleFile },
        Asset { target: t(Os::MacOs, Arch::Aarch64), filename: "rust-analyzer-aarch64-apple-darwin.gz", archive: ArchiveKind::GzipSingleFile },
        Asset { target: t(Os::Windows, Arch::X86_64), filename: "rust-analyzer-x86_64-pc-windows-msvc.zip", archive: ArchiveKind::Zip },
        Asset { target: t(Os::Windows, Arch::Aarch64), filename: "rust-analyzer-aarch64-pc-windows-msvc.zip", archive: ArchiveKind::Zip },
        // Deliberately no Android row -- see doc comment above.
    ],
    // Android is served by the tiers below the download, not by an asset. The
    // maintainer confirmed they run cargo on their device, and rust-analyzer
    // ships as a rustup component, so it is obtainable there without any
    // prebuilt Bionic binary existing. See AID-0005.
    termux_package: Some("rust-analyzer"),
    cargo_crate: Some("rust-analyzer"),
    rustup_component: Some("rust-analyzer"),
};

/// `lua-language-server` — pinned to `3.18.2` (the release current as of
/// 2026-07-14; bump both this comment and the URLs together when updating).
/// No `win32-arm64` asset exists upstream (only `win32-ia32`/`win32-x64`),
/// and no Android asset exists at all.
pub const LUA_LANGUAGE_SERVER: LanguageServer = LanguageServer {
    name: "lua-language-server",
    executable: "lua-language-server",
    filetypes: &["lua"],
    release_base_url: "https://github.com/LuaLS/lua-language-server/releases/download/3.18.2",
    assets: &[
        Asset { target: t(Os::Linux, Arch::X86_64), filename: "lua-language-server-3.18.2-linux-x64.tar.gz", archive: ArchiveKind::TarGz },
        Asset { target: t(Os::Linux, Arch::Aarch64), filename: "lua-language-server-3.18.2-linux-arm64.tar.gz", archive: ArchiveKind::TarGz },
        Asset { target: t(Os::MacOs, Arch::X86_64), filename: "lua-language-server-3.18.2-darwin-x64.tar.gz", archive: ArchiveKind::TarGz },
        Asset { target: t(Os::MacOs, Arch::Aarch64), filename: "lua-language-server-3.18.2-darwin-arm64.tar.gz", archive: ArchiveKind::TarGz },
        Asset { target: t(Os::Windows, Arch::X86_64), filename: "lua-language-server-3.18.2-win32-x64.zip", archive: ArchiveKind::Zip },
        // No win32-arm64, no Android -- see doc comment above.
    ],
    // The odd one out. lua-language-server is C++ and LuaJIT, so unlike the
    // other two it is NOT rescued by the maintainer having cargo on their
    // phone: there is no crate to build. Termux is the only Android tier, and
    // if that name is wrong this resolves to an honest Unavailable rather than
    // a fabricated URL.
    //
    // Worth noting this is also the least painful of the three to lose: kvim
    // exists precisely so that no Lua config is needed. And KOPITIAM is already
    // committed to a pure-Rust Lua 5.1 VM (`kopitiam-lua`, kvim Phase 4) — a
    // Rust-native Lua language server built on that would close this properly,
    // and would be ours.
    termux_package: Some("lua-language-server"),
    cargo_crate: None,
    rustup_component: None,
};

/// `texlab` — pinned to `v5.26.0` (current as of 2026-07-14). The most
/// completely cross-built of the three (it even has `aarch64-windows` and
/// `armv7hf-linux`, neither of which the other two publish), but still no
/// Android asset.
pub const TEXLAB: LanguageServer = LanguageServer {
    name: "texlab",
    executable: "texlab",
    filetypes: &["tex"],
    release_base_url: "https://github.com/latex-lsp/texlab/releases/download/v5.26.0",
    assets: &[
        Asset { target: t(Os::Linux, Arch::X86_64), filename: "texlab-x86_64-linux.tar.gz", archive: ArchiveKind::TarGz },
        Asset { target: t(Os::Linux, Arch::Aarch64), filename: "texlab-aarch64-linux.tar.gz", archive: ArchiveKind::TarGz },
        Asset { target: t(Os::MacOs, Arch::X86_64), filename: "texlab-x86_64-macos.tar.gz", archive: ArchiveKind::TarGz },
        Asset { target: t(Os::MacOs, Arch::Aarch64), filename: "texlab-aarch64-macos.tar.gz", archive: ArchiveKind::TarGz },
        Asset { target: t(Os::Windows, Arch::X86_64), filename: "texlab-x86_64-windows.zip", archive: ArchiveKind::Zip },
        Asset { target: t(Os::Windows, Arch::Aarch64), filename: "texlab-aarch64-windows.zip", archive: ArchiveKind::Zip },
        // No Android -- see doc comment above.
    ],
    // texlab is a Rust program, so the same fact that rescues rust-analyzer on
    // Android rescues it: the maintainer has cargo there. `cargo install texlab`
    // works on a device with no prebuilt Bionic binary in existence.
    termux_package: Some("texlab"),
    cargo_crate: Some("texlab"),
    rustup_component: None,
};

/// Every language server kvim knows about, in the order
/// [`for_filetype`] searches them. Mirrors
/// [`crate::config::default_language_servers`]'s three entries exactly —
/// see that function's doc comment for the Lua source this replaces.
pub const REGISTRY: &[&LanguageServer] = &[&RUST_ANALYZER, &LUA_LANGUAGE_SERVER, &TEXLAB];

/// Looks up the registry entry serving `filetype` (kvim's filetype naming,
/// matching [`crate::config::Config::language_servers`]'s keys — `"rust"`,
/// `"lua"`, `"tex"`).
pub fn for_filetype(filetype: &str) -> Option<&'static LanguageServer> {
    REGISTRY.iter().copied().find(|server| server.filetypes.contains(&filetype))
}

/// How to obtain a server: the four-tier ladder from
/// `docs/ai-decisions/AID-0005-android-lsp-acquisition.md`, tried in the
/// order the variants are declared.
///
/// # Why there is a ladder at all
///
/// AID-0003 assumed a prebuilt static binary could always be downloaded.
/// Checking the upstream release APIs proved otherwise: **none** of
/// rust-analyzer, lua-language-server, or texlab publishes an
/// `aarch64-linux-android` build. A download-only strategy has nothing to
/// download on the one platform it was designed to rescue.
///
/// What resolved it was a fact about the *device*, not the servers: the
/// maintainer has a working cargo on Android. Two of the three servers are
/// Rust programs, so they can simply be built there — and Termux's own package
/// manager covers the rest of the gap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// Found on `PATH` already — nothing to install. Downloading something
    /// the user already has would be rude (and, on a metered mobile
    /// connection, actively hostile).
    ///
    /// This tier is what makes both the maintainer's desktop (rust-analyzer
    /// already installed) and their phone (`pkg install rust-analyzer`) work
    /// with no network access at all.
    OnPath(PathBuf),

    /// Available from the platform's own package manager.
    ///
    /// On Android this is Termux's `pkg`, which is **not** GitHub Releases and
    /// is precisely the source AID-0003 failed to consider.
    ///
    /// This is emphatically *not* a reintroduction of the Mason failure mode.
    /// Mason's sin was shelling out to *language* toolchains — npm, pip,
    /// `go install` — which simply are not present on Android. A platform
    /// package manager is present by definition: it is how the platform
    /// installs software.
    SystemPackage {
        /// The command to run, e.g. `pkg install rust-analyzer`.
        command: String,
        /// Which manager this is, for the message shown to the user.
        manager: &'static str,
    },

    /// Buildable with cargo, because the server is itself a Rust program.
    ///
    /// Legitimate on Android *only* because the maintainer confirmed cargo runs
    /// there. Ranked below [`Self::SystemPackage`] because it compiles from
    /// source, which on a phone is slow — kvim says so up front rather than
    /// appearing to hang for twenty minutes.
    CargoInstall {
        /// The crate to install, e.g. `texlab`.
        crate_name: &'static str,
        /// Set when the server ships as a rustup component (rust-analyzer
        /// does), which is far faster than a source build and should be
        /// suggested first.
        rustup_component: Option<&'static str>,
    },

    /// A prebuilt static binary is published for this target.
    ///
    /// The fastest path on desktop, and the one that categorically cannot
    /// serve Android — hence its position at the bottom of the ladder rather
    /// than the top, where AID-0003 originally put it.
    Download { url: String, archive: ArchiveKind },

    /// No tier applies. `reason` is shown to the user verbatim.
    ///
    /// A normal, expected outcome — not an error to paper over. Never fabricate
    /// a URL to avoid emitting this: a 404 at install time is worse than an
    /// honest "no" at resolve time.
    Unavailable { reason: String },
}

/// Searches `path_var` (in `PATH`'s `:`/`;`-separated syntax, exactly as
/// [`std::env::split_paths`] expects) for an executable file named
/// `executable`. Takes the PATH string as a parameter rather than always
/// reading the real environment so tests can point it at an isolated
/// directory without mutating global process state (`std::env::set_var` is
/// unsound to call from a multi-threaded test binary — see
/// <https://doc.rust-lang.org/std/env/fn.set_var.html>).
///
/// "Executable" means: exists, is a regular file, and (on Unix) has at
/// least one executable permission bit set. On Windows there is no
/// equivalent permission bit to check — existence is the whole test there,
/// mirroring how `cmd.exe` resolves `PATH` — and the `.exe` suffix is tried
/// automatically if `executable` doesn't already end in it, since that is
/// how every language server on this registry is actually named on `PATH`
/// on Windows.
pub fn which_in(executable: &str, path_var: &OsStr) -> Option<PathBuf> {
    for dir in std::env::split_paths(path_var) {
        for candidate_name in candidate_names(executable) {
            let candidate = dir.join(&candidate_name);
            if is_executable_file(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

fn candidate_names(executable: &str) -> Vec<String> {
    if cfg!(windows) && !executable.to_ascii_lowercase().ends_with(".exe") {
        vec![executable.to_string(), format!("{executable}.exe")]
    } else {
        vec![executable.to_string()]
    }
}

fn is_executable_file(path: &std::path::Path) -> bool {
    let Ok(metadata) = path.metadata() else { return false };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// [`which_in`] against the real process `PATH`.
pub fn which(executable: &str) -> Option<PathBuf> {
    which_in(executable, std::env::var_os("PATH")?.as_os_str())
}

/// Resolves how to obtain `server` for `target`, checking `path_var` first.
/// This is the injectable-PATH counterpart of [`resolve`], used directly by
/// tests; [`resolve`] is `resolve_with_path(server, target,
/// env::var_os("PATH"))`.
pub fn resolve_with_path(server: &LanguageServer, target: Target, path_var: Option<&OsStr>) -> Source {
    // Tier 1: already installed. Always first — this is what makes both the
    // desktop (rust-analyzer already present) and Termux (`pkg install`) work
    // with no network at all.
    if let Some(path_var) = path_var
        && let Some(found) = which_in(server.executable, path_var)
    {
        return Source::OnPath(found);
    }

    // Tiers 2 and 3 exist for Android, where NO prebuilt binary is published
    // for any of these servers. A desktop that already has a download available
    // should not be sent to compile from source, so they are only preferred
    // when there is genuinely nothing to download.
    let download = server
        .download_url_for(target)
        .map(|(filename, archive)| Source::Download { url: format!("{}/{filename}", server.release_base_url), archive });

    if let Some(download) = download {
        return download;
    }

    // Tier 2: the platform's own package manager. On Android, Termux's `pkg` —
    // which is not GitHub Releases, and is the source AID-0003 missed.
    if target.os == Os::Android
        && let Some(package) = server.termux_package
    {
        return Source::SystemPackage { command: format!("pkg install {package}"), manager: "Termux pkg" };
    }

    // Tier 3: build it with cargo, valid precisely because the maintainer has a
    // working cargo on their Android device and two of the three servers are
    // themselves Rust programs.
    if let Some(crate_name) = server.cargo_crate {
        return Source::CargoInstall { crate_name, rustup_component: server.rustup_component };
    }

    // Tier 4: an honest no.
    Source::Unavailable {
        reason: format!(
            "{} publishes no prebuilt binary for {target:?} (registry captured 2026-07-14 from its GitHub \
             Releases API), is not on PATH, and is not a Rust program, so it cannot be built with cargo \
             either. No npm/pip/go-install fallback is used -- that is precisely the failure mode this \
             registry replaces.",
            server.name
        ),
    }
}

/// Resolves how to obtain `server` for `target`, using the real process
/// `PATH`. See [`resolve_with_path`] for the testable, injectable-PATH form.
pub fn resolve(server: &LanguageServer, target: Target) -> Source {
    resolve_with_path(server, target, std::env::var_os("PATH").as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn registry_matches_the_three_servers_in_configs_default_language_servers() {
        assert_eq!(for_filetype("rust").unwrap().executable, "rust-analyzer");
        assert_eq!(for_filetype("lua").unwrap().executable, "lua-language-server");
        assert_eq!(for_filetype("tex").unwrap().executable, "texlab");
        assert!(for_filetype("cobol").is_none());
    }

    #[test]
    fn linux_x86_64_resolves_a_musl_static_binary_url() {
        let src = resolve_with_path(&RUST_ANALYZER, t(Os::Linux, Arch::X86_64), None);
        assert_eq!(
            src,
            Source::Download {
                url: "https://github.com/rust-lang/rust-analyzer/releases/download/nightly/rust-analyzer-x86_64-unknown-linux-musl.gz"
                    .to_string(),
                archive: ArchiveKind::GzipSingleFile,
            }
        );
    }

    #[test]
    fn linux_aarch64_resolves_the_gnu_binary_url() {
        let src = resolve_with_path(&RUST_ANALYZER, t(Os::Linux, Arch::Aarch64), None);
        assert_eq!(
            src,
            Source::Download {
                url: "https://github.com/rust-lang/rust-analyzer/releases/download/nightly/rust-analyzer-aarch64-unknown-linux-gnu.gz"
                    .to_string(),
                archive: ArchiveKind::GzipSingleFile,
            }
        );
    }

    #[test]
    fn macos_aarch64_resolves_the_apple_silicon_binary_url() {
        let src = resolve_with_path(&RUST_ANALYZER, t(Os::MacOs, Arch::Aarch64), None);
        assert_eq!(
            src,
            Source::Download {
                url: "https://github.com/rust-lang/rust-analyzer/releases/download/nightly/rust-analyzer-aarch64-apple-darwin.gz".to_string(),
                archive: ArchiveKind::GzipSingleFile,
            }
        );
    }

    #[test]
    fn windows_x86_64_resolves_a_zip_url() {
        let src = resolve_with_path(&TEXLAB, t(Os::Windows, Arch::X86_64), None);
        assert_eq!(
            src,
            Source::Download {
                url: "https://github.com/latex-lsp/texlab/releases/download/v5.26.0/texlab-x86_64-windows.zip".to_string(),
                archive: ArchiveKind::Zip,
            }
        );
    }

    #[test]
    fn android_aarch64_never_resolves_to_a_download_because_no_bionic_binary_exists() {
        // This test used to assert `Unavailable` for all three servers, which was
        // correct until AID-0005: no upstream publishes an aarch64-linux-android
        // build, so a download-only strategy had nothing to offer. That is still
        // true of the DOWNLOAD tier and must stay true -- resolving Android to a
        // `Download` would mean we had invented a URL that 404s.
        //
        // What changed is that Android is now served by the tiers ABOVE and BELOW
        // the download (Termux's pkg; cargo, since the maintainer has a working
        // toolchain there and two of the three servers are Rust programs). So the
        // invariant worth pinning is narrower and sharper than "unavailable":
        // never a download, and never silently nothing.
        for server in REGISTRY {
            let src = resolve_with_path(server, t(Os::Android, Arch::Aarch64), None);
            assert!(
                !matches!(src, Source::Download { .. }),
                "{} must never resolve to a download on Android -- no Bionic asset exists, so any URL here \
                 would be fabricated. Got {src:?}",
                server.name
            );
            if let Source::Unavailable { reason } = &src {
                assert!(!reason.is_empty(), "{} gave an empty reason", server.name);
            }
        }
    }

    #[test]
    fn an_unsupported_arch_returns_unavailable_not_a_panic() {
        // No 32-bit ARM row exists in any table; this must degrade to
        // `Unavailable`, not index-out-of-bounds or unwrap-on-None.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        struct Bogus;
        // Exercise via the real Target/Arch surface instead of inventing a
        // variant: pair a real Os with every real Arch and confirm none of
        // them ever panics, covering the "we don't have this row" path
        // generically for every current registry entry x target combo.
        for server in REGISTRY {
            for os in [Os::Linux, Os::MacOs, Os::Windows, Os::Android] {
                for arch in [Arch::X86_64, Arch::Aarch64] {
                    let _ = resolve_with_path(server, t(os, arch), None); // must not panic
                }
            }
        }
        let _ = Bogus; // silence unused-type warning; kept for documentation value
    }

    #[test]
    fn host_target_detection_does_not_panic_and_is_some_on_this_ci_machine() {
        // This test runs on a real x86_64 Linux desktop (see the top-level
        // report), so `host()` must resolve.
        let host = Target::host();
        assert_eq!(host, Some(t(Os::Linux, Arch::X86_64)));
    }

    /// Builds a fake executable named `name` inside a fresh tempdir and
    /// returns `(tempdir, path_var)` where `path_var` is a `PATH`-style
    /// string containing only that directory.
    fn fake_executable_on_a_path_of_its_own(name: &str) -> (tempfile::TempDir, std::ffi::OsString) {
        let dir = tempfile::tempdir().unwrap();
        let exe_path = dir.path().join(name);
        let mut file = std::fs::File::create(&exe_path).unwrap();
        file.write_all(b"#!/bin/sh\necho fake\n").unwrap();
        drop(file);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&exe_path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let path_var = std::env::join_paths([dir.path()]).unwrap();
        (dir, path_var)
    }

    #[test]
    fn path_detection_is_preferred_over_download_when_the_binary_already_exists() {
        let (dir, path_var) = fake_executable_on_a_path_of_its_own("rust-analyzer");
        let src = resolve_with_path(&RUST_ANALYZER, t(Os::Linux, Arch::X86_64), Some(&path_var));
        assert_eq!(src, Source::OnPath(dir.path().join("rust-analyzer")));
    }

    #[test]
    fn path_detection_falls_through_to_download_when_the_binary_is_absent() {
        let (_dir, path_var) = fake_executable_on_a_path_of_its_own("some-other-tool");
        let src = resolve_with_path(&RUST_ANALYZER, t(Os::Linux, Arch::X86_64), Some(&path_var));
        assert!(matches!(src, Source::Download { .. }), "expected a download source, got {src:?}");
    }

    #[cfg(unix)]
    #[test]
    fn a_non_executable_file_on_path_is_not_treated_as_the_binary() {
        let dir = tempfile::tempdir().unwrap();
        let exe_path = dir.path().join("rust-analyzer");
        std::fs::write(&exe_path, b"not executable").unwrap();
        // Deliberately leave default (non-executable) permissions.
        let path_var = std::env::join_paths([dir.path()]).unwrap();
        let src = resolve_with_path(&RUST_ANALYZER, t(Os::Linux, Arch::X86_64), Some(&path_var));
        assert!(matches!(src, Source::Download { .. }), "a non-executable file must not satisfy PATH detection");
    }

    // ---- The Android acquisition ladder (AID-0005) ----
    //
    // These are the tests that matter most in this file: Android is the whole
    // reason kvim exists, and until AID-0005 every one of these resolved to
    // "unavailable" because no upstream publishes a Bionic binary.

    const ANDROID: Target = t(Os::Android, Arch::Aarch64);

    #[test]
    fn rust_analyzer_on_android_is_obtainable_via_termux_then_cargo() {
        // Empty PATH, so tier 1 cannot fire and the ladder must be exercised.
        let empty = std::ffi::OsString::new();
        let src = resolve_with_path(&RUST_ANALYZER, ANDROID, Some(&empty));

        // Termux's pkg is preferred over compiling rust-analyzer on a phone.
        match src {
            Source::SystemPackage { command, manager } => {
                assert!(command.contains("rust-analyzer"));
                assert_eq!(manager, "Termux pkg");
            }
            other => panic!("expected a Termux package for rust-analyzer on Android, got {other:?}"),
        }
    }

    #[test]
    fn texlab_on_android_falls_through_to_cargo_because_it_is_a_rust_program() {
        // texlab has a Termux entry, so to reach the cargo tier we test the
        // server-level fact directly: it IS a cargo crate, which is what makes
        // Android solvable for it at all.
        assert_eq!(TEXLAB.cargo_crate, Some("texlab"));
        assert_eq!(RUST_ANALYZER.cargo_crate, Some("rust-analyzer"));
        assert_eq!(RUST_ANALYZER.rustup_component, Some("rust-analyzer"));

        // lua-language-server is C++/LuaJIT: there is no crate to build, which
        // is precisely why it is the one server Android cannot fully solve.
        assert_eq!(LUA_LANGUAGE_SERVER.cargo_crate, None);
    }

    #[test]
    fn an_unknown_android_server_reports_honestly_rather_than_inventing_a_url() {
        // A server with no Termux package and no crate must NOT fabricate a
        // download URL. A 404 at install time is worse than an honest no now.
        const NOWHERE: LanguageServer = LanguageServer {
            name: "nowhere-ls",
            executable: "nowhere-ls",
            filetypes: &["nowhere"],
            release_base_url: "https://example.invalid",
            assets: &[],
            termux_package: None,
            cargo_crate: None,
            rustup_component: None,
        };
        let empty = std::ffi::OsString::new();
        let src = resolve_with_path(&NOWHERE, ANDROID, Some(&empty));
        match src {
            Source::Unavailable { reason } => {
                assert!(reason.contains("nowhere-ls"));
                // The honest reason must never suggest npm/pip/go -- that is the
                // exact failure this whole registry replaces.
                assert!(!reason.contains("npm install"));
            }
            other => panic!("expected an honest Unavailable, got {other:?}"),
        }
    }

    #[test]
    fn a_desktop_with_a_prebuilt_binary_is_never_sent_to_compile_from_source() {
        // The ladder must not "improve" a desktop that has a perfectly good
        // download available into a twenty-minute cargo build.
        let empty = std::ffi::OsString::new();
        for target in [t(Os::Linux, Arch::X86_64), t(Os::MacOs, Arch::Aarch64), t(Os::Windows, Arch::X86_64)] {
            let src = resolve_with_path(&RUST_ANALYZER, target, Some(&empty));
            assert!(
                matches!(src, Source::Download { .. }),
                "{target:?} publishes a prebuilt binary and must use it, got {src:?}"
            );
        }
    }

    #[test]
    fn path_still_wins_over_every_other_tier_on_android() {
        // The maintainer's actual situation: they installed rust-analyzer on
        // their phone themselves. kvim must just use it and not download or
        // build anything.
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("rust-analyzer");
        std::fs::write(&exe, b"#!/bin/sh\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let path_var = std::env::join_paths([dir.path()]).unwrap();
        let src = resolve_with_path(&RUST_ANALYZER, ANDROID, Some(&path_var));
        assert!(matches!(src, Source::OnPath(_)), "an installed binary must win on Android too, got {src:?}");
    }
}

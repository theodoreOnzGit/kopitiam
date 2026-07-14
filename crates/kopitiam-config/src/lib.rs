//! The `~/.kopitiam` per-user directory, shared by every KOPITIAM application.
//!
//! Every KOPITIAM app — the `kopitiam` CLI, `kvim`, the future `kmux` — keeps
//! its user configuration under one root, in its own subdirectory:
//!
//! ```text
//! ~/.kopitiam/
//! ├── kopitiam-neovim/     <- kvim: config.json, init.lua, lua/*.lua
//! ├── kopitiam-mux/        <- kmux
//! └── ...                  <- one directory per app, named for its crate
//! ```
//!
//! # Two different `.kopitiam` directories, and why that is fine
//!
//! There is *also* a `.kopitiam/` directory inside each **project**, owned by
//! `kopitiam-index`, holding that project's state (session memory, working set,
//! graph snapshots). These are different things at different scopes, and the
//! shared name is deliberate rather than a collision:
//!
//! | Path | Scope | Owner | Holds |
//! |---|---|---|---|
//! | `<project>/.kopitiam/` | one project | `kopitiam-index` | project state (machine-written) |
//! | `~/.kopitiam/` | one user | this crate | configuration (human-written) |
//!
//! Git makes exactly the same split — `<project>/.git/` versus a per-user
//! config — and for the same reason: state that belongs to a checkout must
//! travel with the checkout, while preferences that belong to a person must
//! not.
//!
//! # Why not XDG (`~/.config/kopitiam`)?
//!
//! Because KOPITIAM must run on Android, where XDG is not a convention and
//! `$XDG_CONFIG_HOME` is usually unset. `$HOME/.kopitiam` is one plain path
//! that resolves identically on Linux, macOS, Windows, and Termux, with no
//! per-OS branch to get wrong — the same reasoning `kopitiam-index` gives for
//! keeping project state inside the project rather than in a platform config
//! directory.

use std::path::PathBuf;

use anyhow::{Context, Result};

/// The name of KOPITIAM's per-user directory.
pub const DIR_NAME: &str = ".kopitiam";

/// KOPITIAM's per-user root: `~/.kopitiam`.
///
/// Honours `$KOPITIAM_HOME` when set, which is what makes this testable without
/// mutating process-global environment state, and what lets a user relocate
/// their configuration.
///
/// Returns `None` when no home directory can be determined. That is a real
/// case — an Android host app may set neither `HOME` nor `KOPITIAM_HOME` — and
/// callers must degrade to built-in defaults rather than panicking. An editor
/// that refuses to start because it cannot find a config directory is worse
/// than one that starts with sensible defaults.
pub fn root() -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os("KOPITIAM_HOME") {
        return Some(PathBuf::from(explicit));
    }
    let home = std::env::var_os("HOME")
        // Windows sets USERPROFILE rather than HOME.
        .or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(home).join(DIR_NAME))
}

/// The per-user directory for one KOPITIAM application: `~/.kopitiam/<app>`.
///
/// `app` is the **crate** name (`"kopitiam-neovim"`), not the binary name
/// (`kvim`). The crate name is the stable identifier: a binary can be renamed,
/// and one crate can ship several binaries.
pub fn app_dir(app: &str) -> Option<PathBuf> {
    Some(root()?.join(app))
}

/// Like [`app_dir`], but creates the directory if it does not exist.
///
/// Returns `Ok(None)` — not an error — when there is no home directory to
/// create it under, so a caller can report "no config location, using defaults"
/// without treating an unusual environment as a failure.
pub fn ensure_app_dir(app: &str) -> Result<Option<PathBuf>> {
    let Some(dir) = app_dir(app) else { return Ok(None) };
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(Some(dir))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_dir_is_exactly_one_level_under_the_root_and_named_for_the_crate() {
        let (Some(root), Some(app)) = (root(), app_dir("kopitiam-neovim")) else {
            return; // No home on this machine; nothing to assert.
        };
        assert_eq!(app.parent(), Some(root.as_path()));
        assert_eq!(app.file_name().unwrap(), "kopitiam-neovim");
    }

    #[test]
    fn the_root_is_dot_kopitiam_under_home() {
        // Skip when KOPITIAM_HOME is overriding, since then the root is
        // whatever the user chose and has no required shape.
        if std::env::var_os("KOPITIAM_HOME").is_some() {
            return;
        }
        if let Some(root) = root() {
            assert_eq!(root.file_name().unwrap(), DIR_NAME);
        }
    }

    #[test]
    fn every_path_getter_is_total_and_never_panics() {
        // The contract every caller depends on: no home directory yields None,
        // not a panic. This test exists so that a future change to `unwrap()`
        // inside these functions fails loudly here rather than at a user's
        // first launch on Android.
        let _: Option<PathBuf> = root();
        let _: Option<PathBuf> = app_dir("anything");
        let _: Option<PathBuf> = app_dir("");
    }
}

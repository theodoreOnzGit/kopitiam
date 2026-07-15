//! The `models` subcommand group: go and get, then check, the local model
//! weights that KOPITIAM's AI layer runs on.
//!
//! # What "acquisition" means here ah
//!
//! KOPITIAM is offline-first (see `CLAUDE.md`, "Offline First"): the runtime
//! rather use a native local model than a cloud one, and "running out of AI
//! tokens should never prevent productive knowledge work." That promise can
//! only hold if the model weights are really sitting on disk already.
//! *Acquisition* is the act of making a catalogued model present and verified
//! inside the local model store, and got exactly two ways it happen:
//!
//! * **Autofetch (`kopitiam models pull <id>`).** KOPITIAM download every
//!   artifact for the model from the URL recorded in its catalog entry, stream
//!   it into the model store, then check each file's SHA-256 against the
//!   catalog. This one is the shiok convenient path, the one most people take.
//!
//! * **Bring-your-own (BYO).** If you already got the weights — copy from
//!   another machine, download by hand, or make locally — you can just drop
//!   each file at the exact path the store is expecting and skip the network
//!   entirely. `kopitiam models path <id>` print out those expected paths, so
//!   the BYO flow is simply "put the file where `path` say, then you done
//!   already." `kopitiam models verify <id>` confirm a BYO file match the
//!   catalog checksum — worth doing precisely because nothing fetched it hor.
//!
//! Both paths end up at the same invariant: a model is *available* when every
//! artifact is at its store path AND every checksum match. `pull` reach that
//! state over the network; BYO reach it by hand; `verify` prove it either way.
//!
//! # Thin-client discipline, must keep
//!
//! Per `CLAUDE.md`'s Architecture section, this file own **no** business logic
//! at all. The catalog data, the on-disk layout, HTTP downloading, and SHA-256
//! verification all live inside the `kopitiam-models` crate. This module here
//! only parse arguments, call into that crate, and format results for a human
//! to read — including the small byte-size formatting helper below, which is a
//! legit presentation concern of a CLI, not business logic lah.

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Args, Subcommand};
use kopitiam_models::{
    Artifact, Catalog, Error as ModelsError, Fetcher, HttpFetcher, ModelSpec, ModelStore,
    ensure_available,
};

/// Options for `kopitiam models`.
///
/// `models` is a command *group*: on its own it does nothing one, it always
/// dispatch to one of the [`ModelsCommand`] actions below.
#[derive(Args, Debug)]
pub struct ModelsArgs {
    #[command(subcommand)]
    command: ModelsCommand,
}

/// The four things you can do with local model weights.
///
/// Each variant map to exactly one function in this file, and each of those
/// functions is a thin call into `kopitiam-models`.
#[derive(Subcommand, Debug)]
enum ModelsCommand {
    /// List every model in the built-in catalog, and whether got already
    /// locally or not.
    ///
    /// Read `kopitiam_models::Catalog::builtin()` for the catalog and check
    /// each entry against the default model store, so the `present?` column
    /// show what is really on disk right now (whether `pull` fetch it or you
    /// dropped it in by hand).
    List,

    /// Go and get a model by downloading and verifying its artifacts
    /// (autofetch).
    ///
    /// This is the network path: it resolve the id in the catalog, then hand
    /// the whole download-and-verify job to
    /// `kopitiam_models::ensure_available`, streaming live progress to the
    /// terminal. If you already got the weights on disk, no need this one —
    /// see `kopitiam models path` for the bring-your-own flow.
    Pull {
        /// Catalog id of the model to go and get (see `kopitiam models list`).
        id: String,
    },

    /// Print the on-disk artifact path(s) for a model id.
    ///
    /// Also doubles up as the bring-your-own guide: these are the exact paths
    /// the store is expecting, so putting each artifact there make the model
    /// available without ever running `pull`. Exit nonzero if the model not
    /// present yet, and point you to `kopitiam models pull`.
    Path {
        /// Catalog id of the model to locate (see `kopitiam models list`).
        id: String,
    },

    /// Check that a present model's artifacts still match their catalog
    /// checksums.
    ///
    /// Useful after a bring-your-own copy, or to catch a corrupted or
    /// truncated download. Hand everything to
    /// `kopitiam_models::ModelStore::verify`.
    Verify {
        /// Catalog id of the model to check (see `kopitiam models list`).
        id: String,
    },
}

/// Runs `kopitiam models`: dispatch to whichever action you chose.
///
/// Return an [`ExitCode`] instead of `()` because `models path` must exit
/// nonzero when a model not present (so it can compose inside shell scripts),
/// which a plain `anyhow::Result<()>` cannot express unless it treat "not
/// present" as an error message. Real failures still propagate as `Err`.
pub fn run(args: ModelsArgs) -> Result<ExitCode> {
    match args.command {
        ModelsCommand::List => {
            list()?;
            Ok(ExitCode::SUCCESS)
        }
        ModelsCommand::Pull { id } => {
            pull(&id)?;
            Ok(ExitCode::SUCCESS)
        }
        ModelsCommand::Path { id } => path(&id),
        ModelsCommand::Verify { id } => verify(&id),
    }
}

/// Implements `kopitiam models list`.
///
/// Print one row per catalogued model with its id, display name, model family
/// ([`kopitiam_models::Architecture`]), license, total download size, and
/// whether it is present in the local store or not.
fn list() -> Result<()> {
    let catalog = Catalog::builtin();
    let store = ModelStore::with_default_root()?;

    // Column widths are chosen wide enough for the current catalog's longest
    // ids and names so the table stay aligned. If a future entry overflow, the
    // row just wrap a bit — still readable, never wrong.
    println!(
        "{:<28} {:<30} {:<8} {:<22} {:>10}  present?",
        "ID", "NAME", "FAMILY", "LICENSE", "SIZE"
    );
    for spec in &catalog {
        let present = if store.is_present(spec) { "yes" } else { "no" };
        println!(
            "{:<28} {:<30} {:<8} {:<22} {:>10}  {}",
            spec.id,
            spec.display_name,
            family_label(&spec.architecture),
            spec.license,
            human_bytes(total_size(spec)),
            present,
        );
    }

    Ok(())
}

/// Implements `kopitiam models pull <id>` — the autofetch path.
///
/// Resolve `id` in the catalog (error out nicely with the list of valid ids if
/// you anyhow-anyhow type a wrong one), then hand the whole download-and-verify
/// job to [`ensure_available`], wiring the progress closure to a live one-line
/// textual indicator. On success, print the verified local path(s).
fn pull(id: &str) -> Result<()> {
    let spec = find_or_explain(id)?;
    let store = ModelStore::with_default_root()?;

    println!("Going to get {} ({})...", spec.display_name, spec.id);

    // Why wrap the fetcher instead of just passing `HttpFetcher::new()`:
    // the frozen `kopitiam-models` contract puts the `progress` closure on
    // `Fetcher::fetch`, and `ensure_available` is the one that call `fetch`
    // — not us. So the ONLY way to get a live progress indicator onto the
    // terminal is to hand `ensure_available` a `Fetcher` that print progress
    // by itself. `ProgressFetcher` do exactly that: it swap in a closure that
    // print downloaded / total bytes, then forward to the real HTTP fetcher.
    let fetcher = ProgressFetcher::new(HttpFetcher::new());
    let acquired = ensure_available(&store, &spec, &fetcher)?;

    println!("Verified already. Local artifact path(s):");
    for path in &acquired.artifact_paths {
        println!("  {}", path.display());
    }

    Ok(())
}

/// A [`Fetcher`] wrapper that print live download progress before forwarding
/// to a real fetcher (in practice [`HttpFetcher`]).
///
/// See [`pull`] for why this exist at all: `ensure_available` owns the call to
/// `Fetcher::fetch`, so the CLI can only observe progress by supplying a
/// fetcher that report it. This is pure presentation — no downloading logic
/// live here, it all belong to the wrapped fetcher.
struct ProgressFetcher<F> {
    inner: F,
}

impl<F: Fetcher> ProgressFetcher<F> {
    /// Wrap `inner` so its downloads print progress to the terminal.
    fn new(inner: F) -> Self {
        Self { inner }
    }
}

impl<F: Fetcher> Fetcher for ProgressFetcher<F> {
    fn fetch(
        &self,
        url: &str,
        dest: &Path,
        progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> std::result::Result<(), ModelsError> {
        // Rewrite the same one line as bytes come in (`\r`, no newline), then
        // print a final newline once the download is settled so the next line
        // of output don't get clobbered.
        let mut printing = |downloaded: u64, total: Option<u64>| {
            match total {
                Some(total) => print!(
                    "\r  {} / {}   ",
                    human_bytes(downloaded),
                    human_bytes(total)
                ),
                None => print!("\r  {}   ", human_bytes(downloaded)),
            }
            let _ = std::io::stdout().flush();
            // Still forward to whatever closure `ensure_available` gave us, so
            // we don't quietly swallow its own progress accounting.
            progress(downloaded, total);
        };
        let result = self.inner.fetch(url, dest, &mut printing);
        println!();
        result
    }
}

/// Implements `kopitiam models path <id>`.
///
/// Print the expected on-disk path of every artifact for `id`. If the model is
/// present, exit success. If it not there, exit [`ExitCode::FAILURE`] after
/// telling you to `pull` it — the same paths it just printed are exactly where
/// a bring-your-own copy should go.
fn path(id: &str) -> Result<ExitCode> {
    let spec = find_or_explain(id)?;
    let store = ModelStore::with_default_root()?;

    for artifact in &spec.artifacts {
        println!("{}", store.artifact_path(&spec, artifact).display());
    }

    if store.is_present(&spec) {
        Ok(ExitCode::SUCCESS)
    } else {
        eprintln!();
        eprintln!(
            "Model '{}' not present yet leh. Run `kopitiam models pull {}` to go and get it,",
            spec.id, spec.id
        );
        eprintln!("or put the artifact(s) at the path(s) above yourself (bring-your-own).");
        Ok(ExitCode::FAILURE)
    }
}

/// Implements `kopitiam models verify <id>`.
///
/// Hand everything to [`ModelStore::verify`]. Report `OK` on success, and on a
/// checksum mismatch surface the offending artifact plus the expected vs.
/// actual digests (exit nonzero). Other errors — e.g. a missing file — just
/// propagate as `Err`.
fn verify(id: &str) -> Result<ExitCode> {
    let spec = find_or_explain(id)?;
    let store = ModelStore::with_default_root()?;

    match store.verify(&spec) {
        Ok(()) => {
            println!(
                "OK: {} ({}) checksums all match the catalog.",
                spec.display_name, spec.id
            );
            Ok(ExitCode::SUCCESS)
        }
        Err(ModelsError::ChecksumMismatch {
            artifact,
            expected,
            actual,
        }) => {
            eprintln!("Checksum don't match for artifact '{artifact}' sia:");
            eprintln!("  expected: {expected}");
            eprintln!("  actual:   {actual}");
            Ok(ExitCode::FAILURE)
        }
        Err(other) => Err(other.into()),
    }
}

/// Look up `id` in the catalog, or return an error that list out every valid
/// id. Shared by `pull`, `path`, and `verify` so the "unknown model" message
/// is the same across all three.
fn find_or_explain(id: &str) -> Result<ModelSpec> {
    match Catalog::find(id) {
        Some(spec) => Ok(spec),
        None => {
            let valid: Vec<String> = Catalog::builtin().into_iter().map(|s| s.id).collect();
            anyhow::bail!(
                "dunno what model id '{id}' is. Valid ids: {}",
                valid.join(", ")
            )
        }
    }
}

/// Add up an [`Artifact`] set's `size_bytes` to get a spec's total download
/// size.
fn total_size(spec: &ModelSpec) -> u64 {
    spec.artifacts.iter().map(|a: &Artifact| a.size_bytes).sum()
}

/// A short label for a model [`kopitiam_models::Architecture`], for the `list`
/// table. Kept damn trivial and presentation-only; the crate own the enum.
fn family_label(arch: &kopitiam_models::Architecture) -> String {
    use kopitiam_models::Architecture::*;
    match arch {
        Qwen2 => "Qwen2".to_string(),
        Llama => "Llama".to_string(),
        Phi3 => "Phi3".to_string(),
        Gemma => "Gemma".to_string(),
        Other(name) => name.clone(),
    }
}

/// Format a byte count into a short human-readable string (e.g. `0.4 GB`).
///
/// Presentation only — it exist so the CLI's tables read naturally, and is on
/// purpose NOT inside `kopitiam-models`, which deal in exact `u64` bytes. Use
/// decimal (SI) units (1 GB = 1000 MB), matching how model download sizes are
/// normally advertised.
fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1000.0 && unit < UNITS.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::human_bytes;

    #[test]
    fn human_bytes_uses_decimal_units() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1_500), "1.5 KB");
        assert_eq!(human_bytes(400_000_000), "400.0 MB");
        assert_eq!(human_bytes(4_000_000_000), "4.0 GB");
    }
}

//! Adapter selection: pick a real on-CPU local model when got one sitting on
//! disk already, otherwise fall back to the deterministic echo stub — never
//! hang, never die, just because no weights are around.
//!
//! # Why this module exists at all
//!
//! Every `kopitiam-workflow` command (`plan`, and later `implement`,
//! `translate`, `review`, ...) has to answer the same question before it can
//! run: *which* [`kopitiam_ai::ModelAdapter`] do I hand to
//! [`kopitiam_workflow::run_workflow`]? Rather than each command re-deciding
//! that (and re-deciding it slightly differently, which is how drift starts),
//! the decision lives here once, and the commands just call
//! [`select_adapter`].
//!
//! The decision honours `CLAUDE.md`'s Offline-First pipeline directly:
//! "existing knowledge, then native Rust, then **local AI**, then cloud AI as
//! the final fallback." When a real `.gguf` is present we take the local-AI
//! rung ([`kopitiam_ai::LocalAdapter`], real on-CPU inference through
//! `kopitiam-runtime`). When nothing is present we drop to
//! [`kopitiam_ai::EchoAdapter`] — the deterministic stub that echoes the
//! assembled context back — so a token-less, network-less machine can still
//! run the whole pipeline and see exactly what *would* have been sent to a
//! model. Running out of weights never stops the work; it only changes who
//! answers.
//!
//! # How the model path get resolved — and why not `ensure_available`
//!
//! This module only ever uses weights **already on disk**. It deliberately
//! does NOT autofetch (that is `kopitiam models pull`'s job, an explicit user
//! action), for two reasons:
//!
//! 1. A workflow command should stay fast and offline — silently downloading
//!    hundreds of MB the first time someone runs `kopitiam plan` would be a
//!    rude surprise sia.
//! 2. `kopitiam-models`' shipped catalog carries **placeholder** sha256s (64
//!    zeros — see `kopitiam_models::Catalog::builtin`), so
//!    `ensure_available` would fetch and then *deliberately* fail the
//!    checksum gate anyway. Calling it here buys nothing but a slow error.
//!
//! Resolution order (first hit wins):
//!
//! 1. **`KOPITIAM_MODEL_GGUF`** — an explicit path to any `.gguf` you already
//!    got. This is the bring-your-own escape hatch: point it wherever, the
//!    store is not consulted at all.
//! 2. The **default catalog model** ([`DEFAULT_MODEL_ID`], overridable with
//!    `KOPITIAM_MODEL`) present in the local [`kopitiam_models::ModelStore`].
//!    "Present" here means the file exists at its store path — we do NOT run
//!    `ModelStore::verify` against it, because the catalog's placeholder
//!    checksums make that gate meaningless today (a real BYO `.gguf` dropped
//!    at the store path would *fail* verify against a 64-zero hash). The real
//!    gate on runnability is [`kopitiam_ai::LocalAdapter::load`] succeeding —
//!    it parses the GGUF, builds the `QwenModel`, and builds the tokenizer,
//!    which is exactly "can we actually run this file." See
//!    `docs/ai-decisions/AID-0029` for the full reasoning on choosing
//!    load-succeeds over checksum-verifies as the selection gate.
//! 3. Nothing on disk → [`kopitiam_ai::EchoAdapter`], with a clear Singlish
//!    note telling the user the two ways to get a real model.

use std::path::PathBuf;

use kopitiam_ai::{EchoAdapter, LocalAdapter, ModelAdapter};
use kopitiam_models::{Catalog, ModelSpec, ModelStore};

/// The catalog id used when nothing is configured. Small (~350 MB),
/// Apache-2.0, the sensible default first pull — matches
/// `kopitiam_models::Catalog::builtin`'s own "sensible default" entry.
pub const DEFAULT_MODEL_ID: &str = "qwen2.5-0.5b-instruct-q4_0";

/// Env var holding an explicit path to a `.gguf` (bring-your-own). Highest
/// priority — when set, the model store is not consulted.
const MODEL_PATH_ENV: &str = "KOPITIAM_MODEL_GGUF";

/// Env var holding a catalog id to use instead of [`DEFAULT_MODEL_ID`].
const MODEL_ID_ENV: &str = "KOPITIAM_MODEL";

/// The outcome of [`select_adapter`]: either a real local model, or the echo
/// stub with a reason we can explain to the user.
///
/// It owns the chosen adapter so the borrow handed to
/// [`kopitiam_workflow::run_workflow`] lives as long as the caller needs it.
/// Get the trait object with [`SelectedAdapter::adapter`], and the
/// human-facing note with [`SelectedAdapter::notice`].
pub enum SelectedAdapter {
    /// A real on-CPU model loaded successfully. `source` is the `.gguf` it
    /// was loaded from, for the "using local model" note.
    ///
    /// The [`LocalAdapter`] is boxed because it is large (it owns the model
    /// weights and tokenizer), and leaving it inline would bloat every
    /// `SelectedAdapter` — including the common `Echo` case — to the size of
    /// a model handle. Boxing keeps the enum small and the two variants
    /// balanced (clippy's `large_enum_variant`).
    Local {
        adapter: Box<LocalAdapter>,
        source: PathBuf,
    },
    /// Falling back to the deterministic stub. `reason` explains why, so the
    /// note can tell the user how to get a real model.
    Echo {
        adapter: EchoAdapter,
        reason: FallbackReason,
    },
}

/// Why [`select_adapter`] fell back to [`EchoAdapter`]. Carries enough to
/// print an actionable Singlish note (which model, where its store path is).
pub enum FallbackReason {
    /// No `.gguf` was found on disk — not at any BYO path, not in the store.
    NoModelOnDisk {
        /// The catalog id we looked for.
        model_id: String,
        /// The exact store path a bring-your-own copy should be dropped at.
        expected_store_path: PathBuf,
    },
    /// A file *was* found, but [`LocalAdapter::load`] rejected it (corrupt,
    /// truncated, wrong format, missing metadata, ...). We keep working via
    /// Echo rather than dying — the whole point of the fallback.
    LoadFailed {
        /// The path we tried to load.
        source: PathBuf,
        /// The loader's error, rendered, so the user can act on it.
        error: String,
    },
}

impl SelectedAdapter {
    /// The chosen adapter as a `&dyn ModelAdapter`, ready to hand to
    /// [`kopitiam_workflow::run_workflow`].
    pub fn adapter(&self) -> &dyn ModelAdapter {
        match self {
            SelectedAdapter::Local { adapter, .. } => adapter.as_ref(),
            SelectedAdapter::Echo { adapter, .. } => adapter,
        }
    }

    /// `true` if a real local model was chosen. Handy for commands that want
    /// to phrase their output differently for real vs. stubbed inference.
    pub fn is_local(&self) -> bool {
        matches!(self, SelectedAdapter::Local { .. })
    }

    /// A human-facing note explaining what got picked and — when it is the
    /// stub — how to get a real model. Written in Singlish to match the
    /// CLI's voice; the technical bits (paths, command names, env vars) stay
    /// exact.
    pub fn notice(&self) -> String {
        match self {
            SelectedAdapter::Local { source, .. } => {
                format!("Using local model on CPU: {}", source.display())
            }
            SelectedAdapter::Echo { reason, .. } => match reason {
                FallbackReason::NoModelOnDisk {
                    model_id,
                    expected_store_path,
                } => format!(
                    "No local model on disk leh — falling back to kopitiam_ai::EchoAdapter \
                     (deterministic stub, echoes the context back, no real inference).\n\
                     To get a real local model, any one of these can do:\n  \
                     - `kopitiam models pull {model_id}`   (autofetch + verify), or\n  \
                     - drop a verified `.gguf` at: {}   (bring-your-own), or\n  \
                     - point KOPITIAM_MODEL_GGUF at any `.gguf` you already got.\n\
                     Echo keep things working offline in the meantime.",
                    expected_store_path.display(),
                ),
                FallbackReason::LoadFailed { source, error } => format!(
                    "Found a model file but cannot load it, so falling back to \
                     kopitiam_ai::EchoAdapter (deterministic stub).\n  \
                     file:  {}\n  \
                     error: {error}\n\
                     Fix or replace that file, or point KOPITIAM_MODEL_GGUF at a good \
                     `.gguf`. Echo keep things working in the meantime.",
                    source.display(),
                ),
            },
        }
    }
}

/// Where on disk to look for a model, decided *before* any loading is
/// attempted. Split out from [`select_adapter`] so the decision is unit-testable
/// without needing real weights: given a store and env inputs, we can assert
/// which path (if any) gets chosen, deterministically.
#[derive(Debug, PartialEq, Eq)]
enum PathChoice {
    /// An explicit bring-your-own path from `KOPITIAM_MODEL_GGUF`. Not checked
    /// for existence here — [`LocalAdapter::load`] will do that, and a bad
    /// path becomes a clean [`FallbackReason::LoadFailed`].
    Explicit(PathBuf),
    /// The default (or `KOPITIAM_MODEL`-selected) catalog model, present in
    /// the store.
    StorePresent(PathBuf),
    /// Nothing on disk. Carries what the note needs to guide the user.
    Absent {
        model_id: String,
        expected_store_path: PathBuf,
    },
}

/// Decide which on-disk path to try, from a store plus the two env inputs.
///
/// Pure and side-effect-free apart from reading the store's filesystem
/// presence check (`ModelStore::is_present`). Env is passed in rather than
/// read here so tests can drive it without touching the process environment.
fn resolve_path(store: &ModelStore, spec: &ModelSpec, byo_path: Option<PathBuf>) -> PathChoice {
    if let Some(path) = byo_path {
        return PathChoice::Explicit(path);
    }

    // Our catalog models are single-artifact (one `.gguf`); take the first.
    // If a future multi-artifact model appears, `LocalAdapter` still loads a
    // single weights file, so the first artifact is the right one to hand it.
    let expected = spec
        .artifacts
        .first()
        .map(|a| store.artifact_path(spec, a))
        .unwrap_or_else(|| store.root().join(&spec.id));

    if store.is_present(spec) {
        PathChoice::StorePresent(expected)
    } else {
        PathChoice::Absent {
            model_id: spec.id.clone(),
            expected_store_path: expected,
        }
    }
}

/// Pick the adapter every workflow command should run against.
///
/// Reads `KOPITIAM_MODEL_GGUF` / `KOPITIAM_MODEL` from the environment,
/// resolves an on-disk `.gguf` via the default [`ModelStore`], and tries
/// [`LocalAdapter::load`]. Any failure to find or load a model falls back to
/// [`EchoAdapter`] — this function never returns `Err` for a *missing* model,
/// because a missing model is the normal offline case, not an error.
///
/// The caller should print [`SelectedAdapter::notice`] so the user knows
/// which rung of the Offline-First pipeline actually answered.
pub fn select_adapter() -> SelectedAdapter {
    let byo_path = std::env::var_os(MODEL_PATH_ENV)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from);
    let model_id =
        std::env::var(MODEL_ID_ENV).unwrap_or_else(|_| DEFAULT_MODEL_ID.to_string());

    // If the store root can't even be resolved (no HOME / XDG_CACHE_HOME) and
    // there's no BYO path either, there is nowhere to look — that's just the
    // no-model case, so say so and echo.
    let store = match ModelStore::with_default_root() {
        Ok(store) => store,
        Err(_) if byo_path.is_none() => {
            return SelectedAdapter::Echo {
                adapter: EchoAdapter,
                reason: FallbackReason::NoModelOnDisk {
                    model_id,
                    expected_store_path: PathBuf::from("<no model cache dir: set HOME or XDG_CACHE_HOME>"),
                },
            };
        }
        // BYO path is set, so a store we can't resolve doesn't matter — build
        // a throwaway store rooted anywhere; resolve_path won't consult it.
        Err(_) => ModelStore::with_root(std::env::temp_dir()),
    };

    // A spec is only needed for the store lookup; with a BYO path we don't
    // need the catalog at all. Fall back to the default spec for the note.
    let spec = Catalog::find(&model_id).unwrap_or_else(|| {
        Catalog::find(DEFAULT_MODEL_ID).expect("the default model id is always in the built-in catalog")
    });

    match resolve_path(&store, &spec, byo_path) {
        PathChoice::Explicit(path) | PathChoice::StorePresent(path) => match LocalAdapter::load(&path)
        {
            Ok(adapter) => SelectedAdapter::Local {
                adapter: Box::new(adapter),
                source: path,
            },
            Err(error) => SelectedAdapter::Echo {
                adapter: EchoAdapter,
                reason: FallbackReason::LoadFailed {
                    source: path,
                    error: format!("{error:#}"),
                },
            },
        },
        PathChoice::Absent {
            model_id,
            expected_store_path,
        } => SelectedAdapter::Echo {
            adapter: EchoAdapter,
            reason: FallbackReason::NoModelOnDisk {
                model_id,
                expected_store_path,
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default model id must always resolve in the built-in catalog —
    /// [`select_adapter`] leans on that as an `expect`.
    #[test]
    fn default_model_id_is_in_the_catalog() {
        assert!(Catalog::find(DEFAULT_MODEL_ID).is_some());
    }

    /// A BYO env path wins outright and is passed through verbatim, without
    /// the store being consulted.
    #[test]
    fn byo_path_is_chosen_verbatim() {
        let store = ModelStore::with_root(std::env::temp_dir().join("kopitiam-test-empty-store"));
        let spec = Catalog::find(DEFAULT_MODEL_ID).unwrap();
        let byo = PathBuf::from("/some/where/my-own.gguf");

        let choice = resolve_path(&store, &spec, Some(byo.clone()));
        assert_eq!(choice, PathChoice::Explicit(byo));
    }

    /// A default model present at its store path resolves to `StorePresent`
    /// pointing exactly at that path. We only need the file to *exist* (the
    /// selection gate is presence, not checksum — see AID-0029), so an empty
    /// placeholder file is enough to prove the decision.
    #[test]
    fn present_store_model_is_chosen() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::with_root(dir.path());
        let spec = Catalog::find(DEFAULT_MODEL_ID).unwrap();

        let artifact = spec.artifacts.first().unwrap();
        let path = store.artifact_path(&spec, artifact);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not a real gguf, just proving presence").unwrap();

        let choice = resolve_path(&store, &spec, None);
        assert_eq!(choice, PathChoice::StorePresent(path));
    }

    /// An empty store with no BYO path resolves to `Absent`, carrying the id
    /// and the exact expected store path the user's note will point them at.
    #[test]
    fn empty_store_resolves_absent_with_a_guiding_path() {
        let dir = tempfile::tempdir().unwrap();
        let store = ModelStore::with_root(dir.path());
        let spec = Catalog::find(DEFAULT_MODEL_ID).unwrap();

        let artifact = spec.artifacts.first().unwrap();
        let expected = store.artifact_path(&spec, artifact);

        let choice = resolve_path(&store, &spec, None);
        assert_eq!(
            choice,
            PathChoice::Absent {
                model_id: spec.id.clone(),
                expected_store_path: expected,
            }
        );
    }

    /// The no-model note must actually be actionable: name the pull command,
    /// the store path, and the BYO env var. This is what a stranded offline
    /// user reads, so it has to carry all three routes.
    #[test]
    fn no_model_notice_tells_the_user_every_route() {
        let selected = SelectedAdapter::Echo {
            adapter: EchoAdapter,
            reason: FallbackReason::NoModelOnDisk {
                model_id: DEFAULT_MODEL_ID.to_string(),
                expected_store_path: PathBuf::from("/home/u/.cache/kopitiam/models/x.gguf"),
            },
        };
        let note = selected.notice();
        assert!(note.contains("kopitiam models pull"));
        assert!(note.contains(DEFAULT_MODEL_ID));
        assert!(note.contains("KOPITIAM_MODEL_GGUF"));
        assert!(note.contains("/home/u/.cache/kopitiam/models/x.gguf"));
        assert!(!selected.is_local());
    }

    /// A bad BYO path must degrade to Echo, not blow up: `select_adapter`
    /// with `KOPITIAM_MODEL_GGUF` pointing at a nonexistent file returns the
    /// stub with a `LoadFailed` reason, never an `Err` or a panic.
    ///
    /// Guarded by a mutex because it mutates process env, which is global.
    #[test]
    fn bad_byo_path_degrades_to_echo_not_a_crash() {
        use std::sync::Mutex;
        static ENV_LOCK: Mutex<()> = Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap();

        // SAFETY: single-threaded within this lock; we restore after.
        unsafe {
            std::env::set_var(MODEL_PATH_ENV, "/does/not/exist/kopitiam-nope.gguf");
        }
        let selected = select_adapter();
        unsafe {
            std::env::remove_var(MODEL_PATH_ENV);
        }

        assert!(!selected.is_local());
        assert!(selected.notice().contains("cannot load it"));
    }
}

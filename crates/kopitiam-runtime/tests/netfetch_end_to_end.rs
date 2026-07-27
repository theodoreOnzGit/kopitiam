//! End-to-end tests against **real model weights**, fetched from the network.
//!
//! # What these prove that the unit tests cannot
//!
//! Everything below `kopitiam-runtime` is already covered by unit tests against
//! *synthetic* GGUF fixtures. Those prove the code is self-consistent. They
//! cannot prove the thing that actually keeps breaking: that a **real** model,
//! as published, survives the whole chain —
//!
//! ```text
//! fetch ──▶ load GGUF ──▶ build tokenizer ──▶ build weights ──▶ generate
//! ```
//!
//! The bug that prompted this file is exactly that gap. A SmolLM2-360M file
//! downloaded fine, verified fine, and then died at the tokenizer with
//! *"byte-level vocab has no single-byte token for byte 0x04"*. Every synthetic
//! fixture passed throughout, because the fixtures were built the way the code
//! expects rather than the way HuggingFace actually ships.
//!
//! # Why each stage is asserted separately
//!
//! The whole diagnostic value is in **which stage** fails. A single
//! `assert!(chat_works())` would have told us nothing we did not already know.
//! So each model reports a [`Stage`], and a failure names the stage and carries
//! the underlying error — the same principle as `models inspect`: a test that
//! only says "broken" is barely better than the bug report.
//!
//! # These do NOT run by default
//!
//! They are gated behind `KOPITIAM_NETFETCH=1` because they need the network and
//! hundreds of MB of disk, which no ordinary `cargo test` run should assume.
//! Without the flag every test here prints what it skipped and passes — a silent
//! skip would let the suite look green while proving nothing.
//!
//! ```bash
//! # everything in the catalog that is chat weights
//! KOPITIAM_NETFETCH=1 cargo test --release -p kopitiam-runtime --test netfetch_end_to_end -- --nocapture
//!
//! # one model only
//! KOPITIAM_NETFETCH=1 KOPITIAM_NETFETCH_ONLY=smollm2-360m-instruct-q8_0 \
//!   cargo test --release -p kopitiam-runtime --test netfetch_end_to_end -- --nocapture
//!
//! # a model that is NOT in the catalog (e.g. a Gemma you dropped in by hand)
//! KOPITIAM_NETFETCH=1 KOPITIAM_NETFETCH_PATHS=/path/to/gemma-3-1b-it-Q4_K_M.gguf \
//!   cargo test --release -p kopitiam-runtime --test netfetch_end_to_end -- --nocapture
//! ```
//!
//! `--nocapture` matters: the per-stage report is the output, and a passing test
//! that hides it wastes the run.
//!
//! Weights land in the ordinary XDG model store and are `.gitignore`d; nothing
//! is written into the worktree.

use std::path::{Path, PathBuf};

use kopitiam_models::{Catalog, ModelSpec, ModelStore};
use kopitiam_runtime::tokenizer_from_gguf;

/// Env flag that turns these on at all.
const GATE: &str = "KOPITIAM_NETFETCH";
/// Restrict the run to one catalog id.
const ONLY: &str = "KOPITIAM_NETFETCH_ONLY";
/// Extra `.gguf` paths (`:`-separated) to exercise alongside the catalog —
/// the escape hatch for a model whose HuggingFace coordinates are not in the
/// catalog yet.
const EXTRA_PATHS: &str = "KOPITIAM_NETFETCH_PATHS";

/// How far a model got before something broke.
///
/// Ordered so `<` means "got less far", which is what makes a stage a useful
/// thing to report rather than just a label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Stage {
    Fetch,
    LoadGguf,
    Tokenizer,
    Weights,
    Generate,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Stage::Fetch => "fetch",
            Stage::LoadGguf => "load-gguf",
            Stage::Tokenizer => "tokenizer",
            Stage::Weights => "weights",
            Stage::Generate => "generate",
        }
    }
}

/// The outcome for one model: how far it got, and why it stopped.
struct Outcome {
    label: String,
    reached: Stage,
    failure: Option<(Stage, String)>,
}

impl Outcome {
    fn ok(&self) -> bool {
        self.failure.is_none()
    }

    fn line(&self) -> String {
        match &self.failure {
            None => format!("  PASS  {:<34} reached {}", self.label, self.reached.name()),
            Some((stage, err)) => format!(
                "  FAIL  {:<34} died at {}: {}",
                self.label,
                stage.name(),
                first_line(err)
            ),
        }
    }
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or(s)
}

fn enabled() -> bool {
    std::env::var(GATE).is_ok_and(|v| !v.is_empty() && v != "0")
}

/// Runs the full chain against one on-disk `.gguf`, stopping at the first
/// failure and reporting where it stopped.
///
/// Deliberately does NOT use `?` to bail out of the test: a Gemma failing must
/// not prevent SmolLM2 being reported, or a single bad model hides the state of
/// every other one.
fn exercise(label: &str, path: &Path) -> Outcome {
    let mut reached = Stage::Fetch;

    let model = match kopitiam_loader::load_model(path) {
        Ok(m) => m,
        Err(e) => {
            return Outcome {
                label: label.into(),
                reached,
                failure: Some((Stage::LoadGguf, e.to_string())),
            };
        }
    };
    reached = Stage::LoadGguf;

    // The stage that actually broke on a real SmolLM2. Built separately from the
    // model — even though `QwenModel::from_loaded_model` could build its own —
    // precisely because it is the historically-failing step and the one whose
    // failure we most need attributed to `tokenizer`, not lumped under `weights`.
    let tokenizer = match tokenizer_from_gguf(&model) {
        Ok(t) => t,
        Err(e) => {
            return Outcome {
                label: label.into(),
                reached,
                failure: Some((Stage::Tokenizer, e.to_string())),
            };
        }
    };
    reached = Stage::Tokenizer;

    // Config + weights + rope in one call. A failure here is genuinely the
    // weights stage (shape mismatch, a tensor the loader cannot dequantize),
    // distinct from the tokenizer failure above.
    let qwen = match kopitiam_runtime::QwenModel::from_loaded_model(&model) {
        Ok(m) => m,
        Err(e) => {
            return Outcome {
                label: label.into(),
                reached,
                failure: Some((Stage::Weights, e.to_string())),
            };
        }
    };
    reached = Stage::Weights;

    match generate_something(&qwen, &tokenizer) {
        Ok(()) => Outcome { label: label.into(), reached: Stage::Generate, failure: None },
        Err(e) => Outcome { label: label.into(), reached, failure: Some((Stage::Generate, e)) },
    }
}

/// Round-trips a prompt through tokenizer + forward pass and insists the model
/// produces *something decodable*.
///
/// Correctness of the text is explicitly NOT asserted — a quantized 360M model
/// answering a prompt has no single right answer, and pinning one would give a
/// test that fails on every upstream requant for no reason. What is asserted is
/// the property that actually regresses silently: the chain runs and yields
/// in-vocabulary, decodable tokens rather than garbage or a panic.
fn generate_something(
    model: &kopitiam_runtime::QwenModel,
    tokenizer: &kopitiam_tokenizer::BpeTokenizer,
) -> Result<(), String> {
    use kopitiam_tokenizer::Tokenizer;

    let prompt = "The capital of France is";
    let ids = tokenizer.encode(prompt).map_err(|e| format!("encoding the prompt: {e}"))?;
    if ids.is_empty() {
        return Err("tokenizer encoded a non-empty prompt to zero tokens".into());
    }

    let round_trip = tokenizer.decode(&ids).map_err(|e| format!("decoding back: {e}"))?;
    if round_trip.trim().is_empty() {
        return Err(format!("prompt {prompt:?} did not survive an encode/decode round trip"));
    }

    let cfg = kopitiam_runtime::GenerationConfig { max_new_tokens: 8, ..Default::default() };
    let out = kopitiam_runtime::generate(model, tokenizer, prompt, &cfg, |_, _| {})
        .map_err(|e| format!("{e}"))?;
    if out.trim().is_empty() {
        return Err("generate produced no text at all".into());
    }
    println!("        prompt {prompt:?} -> {:?}", first_line(&out));
    Ok(())
}

/// Catalog entries that are chat weights (single `.gguf` artifact), honouring
/// `KOPITIAM_NETFETCH_ONLY`.
fn selected_specs() -> Vec<ModelSpec> {
    let only = std::env::var(ONLY).ok().filter(|s| !s.is_empty());
    Catalog::builtin()
        .into_iter()
        .filter(|s| {
            !s.artifacts.is_empty()
                && s.artifacts.iter().all(|a| a.filename.to_ascii_lowercase().ends_with(".gguf"))
        })
        .filter(|s| only.as_ref().is_none_or(|id| &s.id == id))
        .collect()
}

fn extra_paths() -> Vec<PathBuf> {
    std::env::var(EXTRA_PATHS)
        .ok()
        .into_iter()
        .flat_map(|v| v.split(':').map(PathBuf::from).collect::<Vec<_>>())
        .filter(|p| !p.as_os_str().is_empty())
        .collect()
}

#[test]
fn real_models_survive_fetch_load_tokenize_and_generate() {
    if !enabled() {
        println!(
            "SKIPPED: set {GATE}=1 to run (needs network + hundreds of MB of disk).\n\
             Weights go to the XDG model store and are gitignored."
        );
        return;
    }

    let store = ModelStore::with_default_root()
        .expect("model store root (set HOME or XDG_CACHE_HOME)");
    let fetcher = kopitiam_models::HttpFetcher::new();

    let mut outcomes = Vec::new();

    for spec in selected_specs() {
        let label = spec.id.clone();
        // Fetch is its own stage: "cannot even download it" and "downloaded but
        // will not load" are completely different problems and must not be
        // reported as the same failure.
        match kopitiam_models::ensure_available_resolving(&store, &spec, &fetcher, None) {
            Ok(_) => {
                let path = store.artifact_path(&spec, &spec.artifacts[0]);
                outcomes.push(exercise(&label, &path));
            }
            Err(e) => outcomes.push(Outcome {
                label,
                reached: Stage::Fetch,
                failure: Some((Stage::Fetch, e.to_string())),
            }),
        }
    }

    for path in extra_paths() {
        let label = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        if !path.is_file() {
            outcomes.push(Outcome {
                label,
                reached: Stage::Fetch,
                failure: Some((Stage::Fetch, format!("no such file: {}", path.display()))),
            });
            continue;
        }
        outcomes.push(exercise(&label, &path));
    }

    println!("\nnetfetch end-to-end results:");
    for o in &outcomes {
        println!("{}", o.line());
    }

    assert!(
        !outcomes.is_empty(),
        "{GATE} was set but nothing was selected — check {ONLY}/{EXTRA_PATHS}"
    );

    let failed: Vec<&Outcome> = outcomes.iter().filter(|o| !o.ok()).collect();
    assert!(
        failed.is_empty(),
        "{} of {} model(s) did not survive the chain:\n{}",
        failed.len(),
        outcomes.len(),
        failed.iter().map(|o| o.line()).collect::<Vec<_>>().join("\n")
    );
}

#[test]
fn the_gate_is_off_by_default_so_an_ordinary_test_run_needs_no_network() {
    // Guards the property that makes this file safe to have in the tree at all.
    // If the gate ever defaults on, `cargo test` starts downloading hundreds of
    // MB on somebody's metered Termux connection with no warning.
    if std::env::var(GATE).is_err() {
        assert!(!enabled(), "the netfetch gate must be OFF when {GATE} is unset");
    }
}

#[test]
fn an_empty_or_zero_gate_value_does_not_count_as_enabled() {
    // `KOPITIAM_NETFETCH=` (exported but empty) and `=0` are both the shell's
    // usual way of saying "no". Treating either as "yes" would surprise someone
    // trying to turn it off.
    for v in ["", "0"] {
        // SAFETY-adjacent note: single-threaded within this test, and the value
        // is restored immediately; other tests read the gate independently.
        unsafe { std::env::set_var(GATE, v) };
        assert!(!enabled(), "{GATE}={v:?} must not enable the netfetch run");
    }
    unsafe { std::env::remove_var(GATE) };
}
